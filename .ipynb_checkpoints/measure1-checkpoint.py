import cv2
import math
import numpy
import numpy as np
from skimage.metrics import structural_similarity as ssim

def get_image_ssim(original_img, noisy_img, data_range):
    return ssim(original_img, noisy_img, data_range=data_range, multichannel=False)

def get_set_ssim(originalSet, noisySet, img_height=256, img_width=256, win_size=7):

    originalSet = originalSet.reshape(-1, img_height, img_width)
    noisySet = noisySet.reshape(-1, img_height, img_width)
    

    if win_size % 2 == 0:
        raise ValueError("win_size must be an odd number.")
    if win_size > min(img_height, img_width):
        raise ValueError("win_size exceeds the smaller dimension of the images.")
    
    ssim_sum = 0
    for i in range(originalSet.shape[0]):

        ssim_value = ssim(originalSet[i], noisySet[i], data_range=255 if originalSet[i].dtype == np.uint8 else 1, multichannel=False)
        ssim_sum += ssim_value
    
    return ssim_sum / originalSet.shape[0]