import nibabel as nib
import numpy as np
import glob
import os

file_list = glob.glob('./data/ixi-T1/*.nii.gz', recursive=True)
shapes = set()

def z_crop(data,target_z=130):
    z = data.shape[2]
    start = (z-target_z)//2
    return data[:,:,start:start+target_z]

output_folder = './data/crop'
os.makedirs(output_folder, exist_ok=True)

cropshape=set()
for file in file_list:
    img = nib.load(file)
    data = img.get_fdata()
    cropped_data = z_crop(data)
    cropped_img = nib.Nifti1Image(cropped_data, img.affine)
    cropshape.add(cropped_img.header.get_data_shape())
    output_file = os.path.join(output_folder, os.path.basename(file))
    nib.save(cropped_img, output_file)
print("Center cropping:", cropshape)


