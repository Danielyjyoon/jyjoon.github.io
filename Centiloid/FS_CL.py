import os
import nibabel as nib
from tqdm import tqdm
import numpy as np
import pandas as pd

# Define base directory
base = r'C:\Users\byeby\Desktop\test\centiloid\sev_YC_0'

# Initialize data storage list
data_list = []

# Loop through each patient folder
pids = os.listdir(base)
for _, pid in tqdm(enumerate(pids), total=len(pids)):
    
    FILE_DIR = os.path.join(base, pid)
    if os.path.exists(os.path.join(FILE_DIR, 'CT.nii')):
        file_FBB = os.path.join(FILE_DIR, 'rCTPT.nii')
        orig_mask_path = os.path.join(FILE_DIR, 'seg_CT.nii')
        
        # Load original mask
        orig_mask_load = nib.load(orig_mask_path)
        orig_mask = orig_mask_load.get_fdata()
        
        # Define target ROIs
        MRX_ROI_values = [1003, 2003, 1012, 2012, 1014, 2014, 1020, 2020, 1018, 2018, 1019, 2019,
                          1027, 2027, 1028, 2028, 1002, 2002, 1010, 2010, 1023, 2023, 1026, 2026,
                          1008, 2008, 1025, 2025, 1029, 2029, 1031, 2031, 1009, 2009, 1015, 2015,
                          1030, 2030]

        # Define reference ROIs
        WC_values = [7, 8, 46, 47]
        CG_values = [8, 47]
        WCB_values = [7, 8, 46, 47, 173, 174, 175]
        pons_values = [174]
        
        # Create masks for each ROI
        newmask = np.in1d(orig_mask, MRX_ROI_values).reshape(orig_mask.shape).astype(float)
        WC_mask = np.in1d(orig_mask, WC_values).reshape(orig_mask.shape).astype(float)
        CG_mask = np.in1d(orig_mask, CG_values).reshape(orig_mask.shape).astype(float)
        WCB_mask = np.in1d(orig_mask, WCB_values).reshape(orig_mask.shape).astype(float)
        pons_mask = np.in1d(orig_mask, pons_values).reshape(orig_mask.shape).astype(float)
        
        # Load FBB (PET) image and replace NaNs with 0
        FBB = np.array(nib.load(file_FBB).dataobj).astype(float)
        FBB = np.nan_to_num(FBB, nan=0.0)  # Replace NaN with 0
        
        # Apply masks to get ROIs
        MRX_ROI = np.multiply(FBB, newmask)
        WC_ROI = np.multiply(FBB, WC_mask)
        CG_ROI = np.multiply(FBB, CG_mask)
        WCB_ROI = np.multiply(FBB, WCB_mask)
        pons_ROI = np.multiply(FBB, pons_mask)
        
        # Filter non-zero values
        MRX_ROI = MRX_ROI[MRX_ROI != 0]
        WC_ROI = WC_ROI[WC_ROI != 0]
        CG_ROI = CG_ROI[CG_ROI != 0]
        WCB_ROI = WCB_ROI[WCB_ROI != 0]
        pons_ROI = pons_ROI[pons_ROI != 0]
        
        # Calculate SUVR values for each reference region
        SUVR_WC = MRX_ROI.mean() / WC_ROI.mean() if WC_ROI.size > 0 else np.nan
        SUVR_CG = MRX_ROI.mean() / CG_ROI.mean() if CG_ROI.size > 0 else np.nan
        SUVR_WCB = MRX_ROI.mean() / WCB_ROI.mean() if WCB_ROI.size > 0 else np.nan
        SUVR_pons = MRX_ROI.mean() / pons_ROI.mean() if pons_ROI.size > 0 else np.nan
        
        # Append the results to the list
        data_list.append({
            'PID': pid,
            'SUVR_WC': SUVR_WC,
            'SUVR_CG': SUVR_CG,
            'SUVR_WCB': SUVR_WCB,
            'SUVR_pons': SUVR_pons
        })

# Convert the data to a DataFrame
df = pd.DataFrame(data_list)

# Save to Excel file
output_path = 'sevYC0_fs_CT_suvr_multiple_masks.xlsx'
df.to_excel(output_path, index=False)
print(f"Results saved to {output_path}")
