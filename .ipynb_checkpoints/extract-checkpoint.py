import os
import numpy as np
import nibabel as nib
import ants
from tqdm import tqdm
import glob

def extract_data(file_list, output_dir, mask_thresh=0.1, low_percent=5, high_percent=95):
    os.makedirs(output_dir, exist_ok=True)
    
    for file in file_list:
        try:
            img_nib = nib.load(file)
            data = img_nib.get_fdata().astype(np.float32)
            img_ants = ants.from_numpy(data)
            mask = ants.get_mask(img_ants).numpy().astype(bool)
            
            brain_data = data[mask]
            p_low = np.percentile(brain_data, low_percent)
            p_high = np.percentile(brain_data, high_percent)
            data_norm = (np.clip(data, p_low, p_high) - p_low) / (p_high - p_low + 1e-8)
            
            valid_slices = [z for z in range(data_norm.shape[2]) 
                            if np.mean(mask[z,:,:]) >= mask_thresh]
            np.savez_compressed(
                os.path.join(output_dir, os.path.basename(file).replace('.nii.gz', '.npz')),
                data=data_norm,
                valid_slices=valid_slices
            )
            print(f"processing file:{file}")
        except Exception as e:
            print(f"Error: {file} - {e}")

file_list = glob.glob(os.path.join("./data/crop/", "*.nii.gz"))
extract_data(file_list, output_dir="./data/extract")