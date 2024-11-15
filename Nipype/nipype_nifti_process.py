import os
import nipype.interfaces.spm as spm
from nipype.interfaces.spm import Coregister, Normalize12, Smooth

# Define the start path to the images folder inside the data directory
start_path = './images'
#Docker용, 
#Powershell
# docker run -it --rm `
#   -v "C:/Users/byeby/Desktop/test/Dockerdata:/data" `
#   -v "C:/Users/byeby/Desktop/test/Dockerdata/images:/data/images" `
#   -p 8888:8888 `
#   nipype/nipype /bin/bash

# 기본 구조는: C:/Users/byeby/Desktop/test/Dockerdata를 /data로 사용하게 되고
#  그 안에 images 폴더를 /data/images로 사용하게 됨.

# 코드를 data에 넣으면 되고 images에 PID로 넣으면 돌릴 수 있음.

# Check if the path exists, print an error if not found
# if not os.path.exists(start_path):
#     print(f"Error: The start path {start_path} does not exist.")
#     exit()

# Get a list of subdirectories within the start_path (patient directories only)
patient_dirs = [os.path.join(start_path, x) for x in os.listdir(start_path) if os.path.isdir(os.path.join(start_path, x))]

# Print the found directories for debugging
print(f"Found patient directories: {patient_dirs}")

# Iterate through each patient directory
for patient_dir in patient_dirs:
    print(f"Processing patient directory: {patient_dir}")


    wpet_file = os.path.join(patient_dir, 'wPT.nii')
    if os.path.exists(wpet_file):
        print(f"already processed in {patient_dir}, skipping processing...")
        continue

    # Identify MR and FDG NIFTI files (adjust patterns as necessary)
    mr_files = [os.path.join(patient_dir, f) for f in os.listdir(patient_dir) if 'MR' in f and f.endswith('.nii')]
    fdg_files = [os.path.join(patient_dir, f) for f in os.listdir(patient_dir) if 'PT' in f and f.endswith('.nii')]

    # Check if MR and FDG files are found
    if not mr_files or not fdg_files:
        print(f"Missing MR or FDG files in {patient_dir}. Skipping...")
        continue

    # Use the first MR and FDG file found for processing (modify logic if needed)
    mr_file = mr_files[0]
    fdg_file = fdg_files[0]

    # Step 1: Co-registration of FDG to MR
    print(f"Co-registering FDG file: {fdg_file} to MR file: {mr_file}")
    coreg = Coregister()
    coreg.inputs.target = mr_file  # MR as the reference
    coreg.inputs.source = fdg_file  # FDG as the moving image
    coreg.inputs.jobtype = "estwrite"  # Estimate and write transformation

    coreg.run()

    # Step 2: Segmentation of MR
    print(f"Segmenting MR file: {mr_file}")
    seg = spm.Segment()  # Corrected call
    seg.inputs.data = mr_file
    seg.run()

    # Step 3: Normalization
    # Apply the deformation field from MR to FDG (FDG will be the "apply_to_files" input)
    print(f"Normalizing MR file and applying it to FDG file.")
    norm12 = spm.Normalize12()  # Corrected initialization
    norm12.inputs.image_to_align = mr_file  # MR to align
    norm12.inputs.apply_to_files = [fdg_file]  # Apply transformation to FDG, passed as a list
    # norm12.inputs.write_bounding_box = [[-78., -112., -70.], [78., 76., 85.]] #1. 작게 나옴
    norm12.inputs.write_bounding_box = [[-90, -124.5, -72], [90.5, 91, 107]]  # 2. new
    # norm12.inputs.write_bounding_box = [[-90, -124.5, -72], [90.5, 91, 107.0]]  # 2. mni용
    # norm12.inputs.affine_regularization_type = 'mni'
    norm12.inputs.write_voxel_sizes = [2, 2, 2]  # Voxel size stays 2mm
    # norm12.inputs.out_prefix = 'w'  # Prefix for normalized files
    norm12.run()

    # # Generate the normalized FDG file name
    # fdg_norm_file = os.path.join(patient_dir, 'w' + os.path.basename(fdg_file))

    # # Step 4: Smoothing of normalized FDG
    # print(f"Smoothing normalized FDG file: {fdg_norm_file}")
    # smooth = spm.Smooth()
    # smooth.inputs.in_files = fdg_norm_file
    # smooth.inputs.fwhm = [6, 6, 6]  # Full-width half-maximum for smoothing
    # smooth.inputs.out_prefix = 'sw'  # Prefix for smoothed files
    # smooth.run()

    print(f"Completed processing for {patient_dir}")

print("All patient data processed.")
