from skimage.restoration import denoise_nl_means, estimate_sigma
import numpy as np
from my_bm3d import *
import cv2
from skimage.util import img_as_float32
from dipy.denoise.nlmeans import nlmeans
from dipy.denoise.noise_estimate import estimate_sigma as dipy_estimate_sigma


def nlm_denoise_dipy(noisy_image, patch_radius=1, block_radius=3):

    noisy_image = img_as_float32(noisy_image)
    noisy_image = np.clip(noisy_image, 0, None)
    noisy_image = np.stack([noisy_image] * 5, axis=-1)

    sigma_rician = dipy_estimate_sigma(noisy_image, N=4)

    denoised_image = nlmeans(
        noisy_image,
        sigma=sigma_rician * 0.8,
        patch_radius=patch_radius,
        block_radius=block_radius,
        rician=True 
    )
    
    print("Image denoised using DIPY NL-Means (Rician-aware)")
    return denoised_image[:, :, 0]

def rician_optimized_nlmeans(noisy_2d):
    # 构建合法4D输入 (x,y,z=1,diff=1)
    noisy_3d = noisy_2d[:, :, np.newaxis].astype(np.float32)
    
    return nlmeans(noisy_3d, 
                  sigma=estimate_sigma(noisy_3d),
                  patch_radius=1,
                  block_radius=3,
                  rician=True)[:, :, 0]  # 输出恢复2D

def bm3d_denoise(noisy_image):
    noisy_image = noisy_image * 255.0
    denoised = []
    Basic_img = BM3D_1st_step(noisy_image)
    Final_img = BM3D_2nd_step(Basic_img, noisy_image)
    denoised.append(Final_img)
    print("Image denoised using BM3D")

    return numpy.array(denoised)