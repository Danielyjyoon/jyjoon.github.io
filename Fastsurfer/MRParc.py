import sys
import shutil, os
import pandas as pd
from pydicom import dcmread
import seaborn as sns
import numpy as np
import ants

import dicom2nifti
import dicom2nifti.settings as settings
settings.disable_validate_slice_increment()

import pickle
from MR.parcellation import MRI_parcellation

import nibabel as nib
from tqdm import tqdm

import nibabel as nib
from tqdm import tqdm
from conform import conform_MR

base = r'C:\Users\byeby\Desktop\test\Dockerdata\Images'
pids = os.listdir(base)
modality = 'MR'

for _, pid in tqdm(enumerate(pids), total=len(pids)):
    parc_home = r'C:\Users\byeby\Desktop\test\Parcellation\MR'
    FILE_DIR = os.path.join(base, pid)
    file_FBB = os.path.join(FILE_DIR, 'PET.nii')
    file_ST = os.path.join(FILE_DIR, f'{modality}.nii')
    file_ST_256 = os.path.join(FILE_DIR, '256_MR.nii')
    file_ST_seg = os.path.join(FILE_DIR, f'seg_{modality}.nii')
    file_PET_resliced = os.path.join(FILE_DIR, 'rMR_FBB.nii')
    ST_nifti = nib.load(file_ST)

    if not os.path.exists(os.path.join(FILE_DIR, f'256_{modality}.nii')):
            conformed_image = conform_MR(ST_nifti)
            nib.save(conformed_image, file_ST_256)
            print(f"Conformed image saved as {file_ST_256}")

    if not os.path.exists(file_ST_seg) and os.path.exists(file_ST):
        MRI_parcellation(parc_home, file_ST, file_ST_256, file_ST_seg, ST_nifti, {})

    if os.path.exists(file_ST_256) and not os.path.exists(file_PET_resliced):

            src_image = ants.image_read(file_FBB)
            dst_image = ants.image_read(file_ST_256)
            registration = ants.registration(fixed=dst_image, moving=src_image, type_of_transform='Rigid')
          
            ants.image_write(registration['warpedmovout'], file_PET_resliced)