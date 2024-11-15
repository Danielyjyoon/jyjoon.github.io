import os
import nibabel as nib
from tqdm import tqdm
import numpy as np
import pandas as pd

# Define base directory and mask paths
base = r'C:\Users\byeby\Desktop\test\Dockerdata\images'
mask_path = r'C:\Users\byeby\Desktop\test\centiloid\GAAIN\Centiloid_Std_VOI\nifti\2mm'
ctx_dir = os.path.join(mask_path, 'voi_ctx_2mm.nii')
wc_dir = os.path.join(mask_path, 'voi_WhlCbl_2mm.nii')
wcb_dir = os.path.join(mask_path, 'voi_WhlCblBrnStm_2mm.nii')
pons_dir = os.path.join(mask_path, 'voi_Pons_2mm.nii')
cg_dir = os.path.join(mask_path, 'voi_CerebGry_2mm.nii')  # New Cerebellar Gray Matter mask

# Load mask files once outside the loop
ctx = np.array(nib.load(ctx_dir).dataobj)
wc = np.array(nib.load(wc_dir).dataobj)
wcb = np.array(nib.load(wcb_dir).dataobj)
pons = np.array(nib.load(pons_dir).dataobj)
cg = np.array(nib.load(cg_dir).dataobj)

# Initialize data storage list
data_list = []

# Loop through each patient folder
pids = os.listdir(base)
for _, pid in tqdm(enumerate(pids), total=len(pids)):
    FILE_DIR = os.path.join(base, pid)
    file_FBB = os.path.join(FILE_DIR, 'wPT.nii')
    
    # Load FBB (PET) file and replace NaNs with 0
    FBB = np.array(nib.load(file_FBB).dataobj)
    FBB = np.nan_to_num(FBB, nan=0.0)  # Replace NaN with 0
    
    # Apply masks to get ROIs
    CTX_ROI = np.multiply(FBB, ctx)
    WC_ROI = np.multiply(FBB, wc)
    WCB_ROI = np.multiply(FBB, wcb)
    PONS_ROI = np.multiply(FBB, pons)
    CG_ROI = np.multiply(FBB, cg)  # Apply Cerebellar Gray Matter mask
    
    # Filter non-zero values
    CTX_ROI = CTX_ROI[CTX_ROI != 0]
    WC_ROI = WC_ROI[WC_ROI != 0]
    WCB_ROI = WCB_ROI[WCB_ROI != 0]
    PONS_ROI = PONS_ROI[PONS_ROI != 0]
    CG_ROI = CG_ROI[CG_ROI != 0]

    # Calculate SUVR ratios for each mask
    SUVR_WC = CTX_ROI.mean() / WC_ROI.mean() if WC_ROI.size > 0 else np.nan
    SUVR_WCB = CTX_ROI.mean() / WCB_ROI.mean() if WCB_ROI.size > 0 else np.nan
    SUVR_PONS = CTX_ROI.mean() / PONS_ROI.mean() if PONS_ROI.size > 0 else np.nan
    SUVR_CG = CTX_ROI.mean() / CG_ROI.mean() if CG_ROI.size > 0 else np.nan  # SUVR for Cerebellar Gray Matter
    
    # Append result to data list
    data_list.append({
        'PID': pid,
        'SUVR_WC': SUVR_WC,
        'SUVR_WCB': SUVR_WCB,
        'SUVR_PONS': SUVR_PONS,
        'SUVR_CG': SUVR_CG
    })

# Convert the data to a DataFrame
df = pd.DataFrame(data_list)

# Save to Excel file
output_path = 'FMM_Klunk_new.xlsx'
df.to_excel(output_path, index=False)
print(f"Results saved to {output_path}")
