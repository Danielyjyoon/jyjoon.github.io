import os
import nibabel as nib
import sys
sys.path.append(r'C:\Users\byeby\Desktop\test\Parcellation')
from conform import conform_MR  

# Paths
base_dir = r'C:\Users\byeby\Desktop\test\Freesurfer\160216'
input_file = os.path.join(base_dir, 'MR.nii')
output_file = os.path.join(base_dir, '256_MR.nii')

# Check if input exists
if os.path.exists(input_file):
    # Load the original MR image
    MR_img = nib.load(input_file)
    
    # Conform the MR image using conform_MR from conform.py
    conformed_MR = conform_MR(MR_img)
    
    # Save the conformed image as 256_MR.nii
    nib.save(conformed_MR, output_file)
    print(f"Conformed MR saved as {output_file}")
else:
    print(f"Input file {input_file} does not exist!")