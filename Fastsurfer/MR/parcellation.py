import argparse
import numpy as np
import time
import torch
import torch.nn as nn

from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms

from scipy.ndimage.filters import median_filter, gaussian_filter
from skimage.measure import label, regionprops
from skimage.measure import label

from collections import OrderedDict

from .data_loader.load_neuroimaging_data import OrigDataThickSlices
from .data_loader.load_neuroimaging_data import map_label2aparc_aseg
from .data_loader.load_neuroimaging_data import map_prediction_sagittal2full
from .data_loader.load_neuroimaging_data import get_largest_cc
from .data_loader.load_neuroimaging_data import load_and_conform_image
from .data_loader.load_neuroimaging_data import save_image

from .data_loader.augmentation import ToTensorTest_MR

from models.networks import FastSurferCNN

def options_parse(parc_home):
    parser = argparse.ArgumentParser()

    parser.add_argument('--order', dest='order', type=int, default=1,
                        help="order of interpolation (0=nearest,1=linear(default),2=quadratic,3=cubic)")

    # 4. Pre-trained weights
    parser.add_argument('--network_sagittal_path', dest='network_sagittal_path',
                        help="path to pre-trained weights of sagittal network",
                        default=f'{parc_home}/weights/sagittal.pkl')
    parser.add_argument('--network_coronal_path', dest='network_coronal_path',
                        help="pre-trained weights of coronal network",
                        default=f'{parc_home}/weights/coronal.pkl')
    parser.add_argument('--network_axial_path', dest='network_axial_path',
                        help="pre-trained weights of axial network",
                        default=f'{parc_home}/weights/axial.pkl')

    # 5. Options for model parameters setup (only change if model training was changed)
    parser.add_argument('--num_filters', type=int, default=128,
                        help='Filter dimensions for DenseNet (all layers same). Default=71')
    parser.add_argument('--num_filters_interpol', type=int, default=32,
                        help='Filter dimensions for DenseNet (all layers same). Default=32')
    parser.add_argument('--num_classes_ax_cor', type=int, default=81)
    parser.add_argument('--num_classes_sag', type=int, default=53)

    parser.add_argument('--num_channels', type=int, default=7,
                        help='Number of input channels. Default=7 (thick slices)')
    parser.add_argument('--kernel_height', type=int, default=5, help='Height of Kernel (Default 3)')
    parser.add_argument('--kernel_width', type=int, default=5, help='Width of Kernel (Default 3)')
    parser.add_argument('--height', type=int, default=256, help='Height (Default 256)')
    parser.add_argument('--width', type=int, default=256, help='Width (Default 256)')
    parser.add_argument('--base_res', type=float, default=1.0, help='Width of Kernel (Default 3)')
    parser.add_argument('--stride', type=int, default=1, help="Stride during convolution (Default 1)")
    parser.add_argument('--stride_pool', type=int, default=2, help="Stride during pooling (Default 2)")
    parser.add_argument('--pool', type=int, default=2, help='Size of pooling filter (Default 2)')

    # 6. Clean up and GPU/CPU options (disable cuda, change batchsize)
    parser.add_argument('--clean', dest='cleanup', help="Flag to clean up segmentation", action='store_true', default=True)
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size for inference. Default: 8")

    sel_option = parser.parse_args(args=[])

    return sel_option

def run_network(img_filename, orig_data, prediction_probability, plane, ckpts, params_model, model):

    # Set up DataLoader
    test_dataset = OrigDataThickSlices(img_filename, orig_data, plane=plane,
                                       transforms=transforms.Compose([ToTensorTest_MR()]))

    test_data_loader = DataLoader(dataset=test_dataset, shuffle=False,
                                  batch_size=params_model["batch_size"])
    
    model_state = torch.load(ckpts, map_location=params_model["device"])
    new_state_dict = OrderedDict()

    for k, v in model_state["model_state_dict"].items():
        if k[:7] == "module." and not params_model["model_parallel"]:
            new_state_dict[k[7:]] = v
        elif k[:7] != "module." and params_model["model_parallel"]:
            new_state_dict["module." + k] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    # model.load_state_dict(model_state["model_state"])
    # model.load_state_dict(new_state_dict)

    model.eval()

    with torch.no_grad():

        start_index = 0
        for batch_idx, sample_batch in enumerate(test_data_loader):

            images_batch = Variable(sample_batch["image"])

            if params_model["use_cuda"]:
                images_batch = images_batch.cuda()

            temp = model(images_batch)

            if plane == "Axial":
                temp = temp.permute(3, 0, 2, 1)
                prediction_probability[:, start_index:start_index + temp.shape[1], :, :] += torch.mul(temp, 0.4)
                start_index += temp.shape[1]

            elif plane == "Coronal":
                temp = temp.permute(2, 3, 0, 1)
                prediction_probability[:, :, start_index:start_index + temp.shape[2], :] += torch.mul(temp, 0.4)
                start_index += temp.shape[2]

            else:
                temp = map_prediction_sagittal2full(temp).permute(0, 3, 2, 1)
                prediction_probability[start_index:start_index + temp.shape[0], :, :, :] += torch.mul(temp, 0.2)
                start_index += temp.shape[0]

    return prediction_probability
   

def MRI_parcellation(parc_home, file_ST, file_ST_256, file_ST_seg, ST_nifti, parc_results):


    torch.cuda.empty_cache()
    # Command Line options and error checking done here
    options = options_parse(parc_home)

    # check what device to use and how much memory is available (memory can be overwritten)
    use_cuda = torch.cuda.is_available()

    # Start Job
    start_total = time.time()
    header_info, affine_info, orig_data = load_and_conform_image(file_ST, ST_nifti, 1)

    # Save conformed MR input image
    parc_results['ST']= save_image(orig_data, affine_info, header_info, file_ST_256, True)
    

    # Set up model for axial and coronal networks
    params_network = {'num_channels': options.num_channels, 'num_filters': options.num_filters,
                      'num_filters_interpol': options.num_filters_interpol,
                    'kernel_h': options.kernel_height, 'kernel_w': options.kernel_width,
                    'height': options.height, 'width': options.width,
                    'stride_conv': options.stride, 'pool': options.pool,
                    'stride_pool': options.stride_pool, 'num_classes': options.num_classes_ax_cor,
                    'kernel_c': 1, 'kernel_d': 1}

    # Select the model
    model = FastSurferCNN(params_network)

    # Put it onto the GPU or CPU
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        model_parallel = True
    else:
        model_parallel = False

    print(model_parallel, 'model parallel')
    model.to(device)

    params_model = {'device': device, "use_cuda": use_cuda, "batch_size": options.batch_size,
                    "model_parallel": model_parallel}

    # Set up tensor to hold probabilities
    pred_prob = torch.zeros((256, 256, 256, options.num_classes_ax_cor), dtype=torch.float).to(device)

    # Axial Prediction
    start = time.time()
    pred_prob = run_network(file_ST,
                            orig_data, pred_prob, "Axial",
                            options.network_axial_path,
                            params_model, model)

    # Coronal Prediction
    start = time.time()
    pred_prob = run_network(file_ST,
                            orig_data, pred_prob, "Coronal",
                            options.network_coronal_path,
                            params_model, model)

    # Sagittal Prediction
    start = time.time()
    params_network["num_classes"] = options.num_classes_sag
    params_network["num_channels"] = options.num_channels

    model = FastSurferCNN(params_network)

    if model_parallel:
        model = nn.DataParallel(model)

    model.to(device)

    pred_prob = run_network(file_ST, orig_data, pred_prob, "Sagittal",
                            options.network_sagittal_path,
                            params_model, model)

    # Get predictions and map to freesurfer label space
    _, pred_prob = torch.max(pred_prob, 3)
    pred_prob = pred_prob.cpu().numpy()
    pred_prob = map_label2aparc_aseg(pred_prob)

    # Post processing - Splitting classes
    # Quick Fix for 2026 vs 1026; 2029 vs. 1029; 2025 vs. 1025
    rh_wm = get_largest_cc(pred_prob == 41)
    lh_wm = get_largest_cc(pred_prob == 2)
    rh_wm = regionprops(label(rh_wm, background=0))
    lh_wm = regionprops(label(lh_wm, background=0))
    centroid_rh = np.asarray(rh_wm[0].centroid)
    centroid_lh = np.asarray(lh_wm[0].centroid)

    labels_list = np.array([1003, 1006, 1007, 1008, 1009, 1011,
                            1015, 1018, 1019, 1020, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1034, 1035])

    for label_current in labels_list:

        label_img = label(pred_prob == label_current, connectivity=3, background=0)

        for region in regionprops(label_img):

            if region.label != 0:  # To avoid background

                if np.linalg.norm(np.asarray(region.centroid) - centroid_rh) < np.linalg.norm(
                        np.asarray(region.centroid) - centroid_lh):
                    mask = label_img == region.label
                    pred_prob[mask] = label_current + 1000

    # Quick Fixes for overlapping classes
    aseg_lh = gaussian_filter(1000 * np.asarray(pred_prob == 2, dtype=float), sigma=3)
    aseg_rh = gaussian_filter(1000 * np.asarray(pred_prob == 41, dtype=float), sigma=3)

    lh_rh_split = np.argmax(np.concatenate((np.expand_dims(aseg_lh, axis=3), np.expand_dims(aseg_rh, axis=3)), axis=3),
                            axis=3)

    # Problematic classes: 1026, 1011, 1029, 1019
    for prob_class_lh in [1011, 1019, 1026, 1029]:
        prob_class_rh = prob_class_lh + 1000
        mask_lh = ((pred_prob == prob_class_lh) | (pred_prob == prob_class_rh)) & (lh_rh_split == 0)
        mask_rh = ((pred_prob == prob_class_lh) | (pred_prob == prob_class_rh)) & (lh_rh_split == 1)

        pred_prob[mask_lh] = prob_class_lh
        pred_prob[mask_rh] = prob_class_rh

    # Clean-Up
    if options.cleanup is True:

        labels = [2, 4, 5, 7, 8, 10, 11, 12, 13, 14,
                15, 16, 17, 18, 24, 26, 28, 31, 41, 43, 44,
                46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 63,
                77, 1026, 2026]

        start = time.time()
        pred_prob_medfilt = median_filter(pred_prob, size=(3, 3, 3))
        mask = np.zeros_like(pred_prob)
        tolerance = 25

        for current_label in labels:
            current_class = (pred_prob == current_label)
            label_image = label(current_class, connectivity=3)

            for region in regionprops(label_image):

                if region.area <= tolerance:
                    mask_label = (label_image == region.label)
                    mask[mask_label] = 1

        pred_prob[mask == 1] = pred_prob_medfilt[mask == 1]
        

    # Saving label
    header_info.set_data_dtype(np.int16)

    parc_results['LABEL']= save_image(pred_prob, affine_info, header_info, file_ST_seg, True)

    torch.cuda.empty_cache()
