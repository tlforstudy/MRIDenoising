from keras.models import load_model
import argparse
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import glob
import os
import dataset_reader
import samples_plt
import h5py
from CNN_denoiser import CNN_denoiser
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot
from autoencoder import *
import conv_denoiser
import measure1
from skimage.transform import resize
import nibabel as nib
from dataset_reader import add_gaussian_noise,add_rician_noise
from skimage.draw import polygon
from skimage.metrics import structural_similarity as ssim
import ants
from scipy import ndimage as ndi
from tensorflow import image as tfimage
import tensorflow as tf
from bm3d import bm3d
from dipy.denoise.noise_estimate import estimate_sigma as dipy_estimate_sigma
from autoencoder import get_autoencoder_model_upsample,get_autoencoder_model_transpose
from skimage.restoration import estimate_sigma
from skimage.util import img_as_float32
import time

def save_samples_compare(noisy_input_test,pure_test, img_height=256, img_width=256):
    plt.style.use('classic')
    fig, axes = plt.subplots(1, 4)
    fig.set_size_inches(25, 10)
    fig.subplots_adjust(wspace=0.15,left=0.03, right=0.97,
        top=0.85, bottom=0.15) 
    axes[0].set_title('Original image')
    axes[1].set_title('Rician noise and Gaussian Blur')
    axes[2].set_title('Rician noise')
    axes[3].set_title('Gaussian Blur')

    for i in range(0, len(noisy_input_test)):
        # Get the sample and the reconstruction
        noisy_image = noisy_input_test[i][:, :, 0]
        pure_image = pure_test[i][:, :, 0]
        denoised_image1 = add_rician_noise(pure_image)
        denoised_image2 = add_gaussian_blur(pure_image)

        axes[0].imshow(pure_image, pyplot.cm.gray)
        axes[0].set_xlabel("SSIM: {:.5f}".format(measure1.get_image_ssim(pure_image, pure_image,data_range=1.0)))
        axes[1].imshow(noisy_image, pyplot.cm.gray)
        axes[1].set_xlabel("SSIM: {:.5f} PSNR: {:.5f}".format(measure1.get_image_ssim(pure_image, noisy_image,data_range=1.0),PSNR(pure_image, noisy_image,max_val=1.0)))
        axes[2].imshow(denoised_image1, pyplot.cm.gray)
        axes[2].set_xlabel("SSIM: {:.5f} PSNR: {:.5f}".format(measure1.get_image_ssim(pure_image, denoised_image1,data_range=1.0),PSNR(pure_image, denoised_image1,max_val=1.0)))
        axes[3].imshow(denoised_image2, pyplot.cm.gray)
        axes[3].set_xlabel("SSIM: {:.5f} PSNR: {:.5f}".format(measure1.get_image_ssim(pure_image, denoised_image2,data_range=1.0),PSNR(pure_image, denoised_image2,max_val=1.0)))


        
        fig.suptitle((
            "Rician noise σ=0.05, Gaussian Blur σ=0.8 kernel=5"),
            fontsize=20, fontweight='bold')

        plt.savefig("results/Compare{0}.png".format(i))

    
    

def DSSIM(y_true, y_pred):
    return tfmath.divide(tfmath.subtract(1.0,tfimage.ssim(y_true, y_pred, max_val=1.0)),2.0)

def PSNR(y_true, y_pred, max_val=1.0):
    y_true = np.expand_dims(y_true, axis=(0, -1))
    y_pred = np.expand_dims(y_pred, axis=(0, -1))
    psnr =tfimage.psnr(y_true, y_pred, max_val).numpy()
    return float(psnr[0])

def get_set_psnr(originalSet, noisySet, data_range=1.0):
    originalSet = originalSet.reshape(-1, img_height, img_width)
    noisySet = noisySet.reshape(-1, img_height, img_width)
    
    original_tensor = tf.convert_to_tensor(originalSet)
    noisy_tensor = tf.convert_to_tensor(noisySet)
    

    if original_tensor.ndim == 3:
        original_tensor = tf.expand_dims(original_tensor, axis=0)
        noisy_tensor = tf.expand_dims(noisy_tensor, axis=0)
    

    if original_tensor.shape != noisy_tensor.shape:
        raise ValueError(f"Not match: original {original_tensor.shape} vs noisy {noisy_tensor.shape}")
    
 
    psnr_values = tf.image.psnr(original_tensor, noisy_tensor, max_val=data_range)
    

    return tf.reduce_mean(psnr_values).numpy()

def add_gaussian_noise_snr(pure, snr=25.0):
    signal_power = np.mean(pure ** 2)
    noise_std = np.sqrt(signal_power / snr)
    noise = np.random.normal(0, noise_std, pure.shape)
    noisy = pure + noise
    
    return np.clip(noisy, 0, 1)

def add_gaussian_blur(pure, sigma=0.8):
    kernel_size = 5
    blurred = cv2.GaussianBlur(pure, (kernel_size, kernel_size), sigma)
    
    return blurred



def read_ixi_t1_x(file_list,batch_size=10,max_attempts=5,slice_thresh=0.1,low_percent=5,high_percent=95):
    num = len(file_list)
    while True:
        for i in range(0, num, batch_size):   
            batchfile = file_list[i:i+batch_size]
            noisy_batch, clean_batch=[] , []
            for file in batchfile:
                if not os.path.exists(file):
                        print(f"File not exist: {file}")
                        continue
                try:
                    img = nib.load(file)
                    data = img.get_fdata().astype(np.float32)
                    p_low = np.percentile(data, low_percent)
                    p_high = np.percentile(data, high_percent)
                    data = np.clip(data, p_low, p_high)
                    data = (data - p_low) / (p_high - p_low + 1e-8)
                    data = resize(data, (256, 256,256), anti_aliasing=True)

                    valid_slice_found = False
                    for _ in range(max_attempts):

                        slice_idx = np.random.randint(0, data.shape[0])
                        clean_slice = data[slice_idx, :, :]

                        slice_nonzero = np.mean(clean_slice > 0.01)
                        if slice_nonzero >= slice_thresh:
                            valid_slice_found = True
                            break
                    
                    if not valid_slice_found:
                        if verbose:
                            print(f"Can't find valid slice: {file} (attempts times: {max_attempts})")
                        skipped_files +=1
                        continue
                    
                    slice_idx = np.random.choice(valid_slices)
                    noisy_data = add_gaussian_noise_snr(data)
                    #noisy_data = add_gaussian_noise(data)
                    #slice_idx = np.random.randint(0, data.shape[0]) #Random slice
                    noisy_slice = noisy_data[slice_idx, :, :]
                    clean_slice = data[slice_idx, :, :]
                    pure = clean_slice.copy()
                    noisy_slice = np.expand_dims(noisy_slice, axis=-1)  # (H, W, 1)
                    clean_slice = np.expand_dims(clean_slice, axis=-1)
                    noisy_slice = np.clip(noisy_slice,0,1)
                    clean_slice = np.clip(clean_slice,0,1)
                    noisy_batch.append(noisy_slice)
                    clean_batch.append(clean_slice)
                    pure.append(pure)
                    
                except Exception as e:
                        print(f"Error: {file}, information: {e}")
                if len(noisy_batch) == 0:
                    continue
            
            yield np.array(noisy_batch), np.array(clean_batch),np.array(pure)

def read_ixi_t1_y(file_list,batch_size=10,max_attempts=5,slice_thresh=0.1,low_percent=5,high_percent=95,mask_thresh=0.1):
    num = len(file_list)
    while True:
        for i in range(0, num, batch_size):   
            batchfile = file_list[i:i+batch_size]
            noisy_batch, clean_batch=[] , []
            for file in batchfile:
                if not os.path.exists(file):
                        print(f"File not exist: {file}")
                        continue
                try:
                    img_nib = nib.load(file)
                    data = img_nib.get_fdata().astype(np.float32)
                    img_ants = ants.from_numpy(data)
                    mask = ants.get_mask(img_ants).numpy().astype(bool)
            
                    brain_data = data[mask]
                    p_low = np.percentile(brain_data, low_percent)
                    p_high = np.percentile(brain_data, high_percent)
                    data_norm = (np.clip(data, p_low, p_high) - p_low) / (p_high - p_low + 1e-8)
                    slice_idx = np.random.randint(0, data.shape[1])
                    noisy_data = add_gaussian_blur(data)
                    #noisy_data = add_gaussian_noise_snr(data)
                    noisy_data = add_gaussian_noise(noisy_data)
                    #slice_idx = np.random.randint(0, data.shape[0]) #Random slice
                    noisy_slice = noisy_data[:, slice_idx, :]
                    clean_slice = data[:, slice_idx,:]
                    noisy_slice = np.expand_dims(noisy_slice, axis=-1)  # (H, W, 1)
                    clean_slice = np.expand_dims(clean_slice, axis=-1)
                    noisy_slice = np.clip(noisy_slice,0,1)
                    clean_slice = np.clip(clean_slice,0,1)
                    noisy_slice = resize(noisy_slice, (128, 128), anti_aliasing=True)
                    clean_slice = resize(clean_slice, (128, 128), anti_aliasing=True)
                    noisy_batch.append(noisy_slice)
                    clean_batch.append(clean_slice)
                    
                    
                
                
                except Exception as e:
                        print(f"Error: {file}, information: {e}")
                        
            current_size = len(noisy_batch)
            if len(noisy_batch) == 0:
                continue
                    
            if current_size < batch_size:
                needed = batch_size - current_size
                indices = np.random.choice(range(current_size), needed)
                for idx in indices:
                    noisy_batch.append(noisy_batch[idx].copy())
                    clean_batch.append(clean_batch[idx].copy())
            print('output data')
            yield np.array(noisy_batch), np.array(clean_batch)

            
            
            
def read_ixi_t1_z(file_list,batch_size=10,slices_per_file=5):
    num = len(file_list)
    while True:
        file_list = np.random.permutation(file_list)
        for i in range(0, num, batch_size):   
            batchfile = file_list[i:i+batch_size]
            noisy_batch, clean_batch=[] , []
            for file in batchfile:
                if not os.path.exists(file):
                        print(f"Not exist: {file}")
                        continue
                try:
                    npz = np.load(file)
                    data = npz['data']
                    valid_slices = npz['valid_slices']
                    selected_slices = np.random.choice(valid_slices, slices_per_file, replace=False)
                    #slice_idx = np.random.randint(0, data.shape[2]) #Random slice
                    for slice_idx in selected_slices:
                        clean_slice = data[:, :, slice_idx]
                        noisy_slice = clean_slice.copy()
                        noisy_slice = add_gaussian_blur(clean_slice.copy())
                        #noisy_slice = add_gaussian_noise(noisy_slice, noise_mean=noise_mean, noise_std=noise_std, noise_prop=noise_prop)
                        noisy_slice = add_rician_noise(noisy_slice,sigma=0.05)
                        noisy_slice = np.expand_dims(noisy_slice, axis=-1)  # (H, W, 1)
                        clean_slice = np.expand_dims(clean_slice, axis=-1)
                        noisy_slice = np.clip(noisy_slice,0,1)
                        clean_slice = np.clip(clean_slice,0,1)

                        noisy_batch.append(noisy_slice)
                        clean_batch.append(clean_slice)
                    
                
                
                except Exception as e:
                        print(f"Error: {file}, information: {e}")
                        
            current_size = len(noisy_batch)
            if len(noisy_batch) == 0:
                continue
                    
            if current_size < batch_size:
                needed = batch_size - current_size
                indices = np.random.choice(range(current_size), needed)
                for idx in indices:
                    noisy_batch.append(noisy_batch[idx].copy())
                    clean_batch.append(clean_batch[idx].copy())
            
            yield np.array(noisy_batch), np.array(clean_batch)

noise_prop = 0.1
noise_std = 1
noise_mean = 0    


number_of_samples = 10
img_width = 256
img_height = 256

custom_objects_rician = {

    'CBAM': CBAM,
    'ChannelAttention': ChannelAttention,
    'SpatialAttention': SpatialAttention,
    'edge_loss': edge_loss,
    'mixed_loss': mixed_loss,
    'mix_loss': mix_loss,
    'ssim' :ssim
}

custom_objects2 = {

    'DSSIM': DSSIM,
    'PSNR': PSNR
}

custom_objects_1 = {

    'CBAM': CBAM,
    'ChannelAttention': ChannelAttention,
    'SpatialAttention': SpatialAttention,
    'edge_loss': edge_loss,
    'mixed_loss': mixed_loss,
    'mix_loss': mix_loss
}

file_list = sorted(glob.glob(os.path.join("./data/extract/", "*.npz")))

#file_list =sorted(glob.glob(os.path.join("./data/crop/", "*.nii.gz")))

train_files, val_files = train_test_split(file_list, test_size=0.2, random_state=42)

val_files, test_files =  train_test_split(val_files, test_size=0.5, random_state=42)

batch_size = 10
steps_per_epoch = len(train_files) // batch_size
validation_steps = len(val_files) // batch_size

#model_res = load_model('./models/res_skip_cae_Rician256.h5',custom_objects=custom_objects2)
#model_res = get_autoencoder_model256()










#test_generator = read_ixi_t1_y(file_list=val_files, batch_size=10)
#test_generator = read_ixi_t1_x(file_list=val_files, batch_size=10)
test_generator = read_ixi_t1_z(file_list=test_files, batch_size=10)

noisy_batch, clean_batch = next(test_generator)


samples = noisy_batch[:number_of_samples]
pure_test = clean_batch[:number_of_samples]

save_samples_compare(samples,pure_test, img_height=256, img_width=256)

def bm3d_denoise_rician_dipy(noisy_image):

    noisy_image = img_as_float32(noisy_image)
    noisy_image = np.clip(noisy_image, 0, None)
    noisy_image = noisy_image[:,:,np.newaxis]
    sigma_rician = dipy_estimate_sigma(noisy_image, N=4)
    

    noisy_scaled = (noisy_image * 255.0).astype(np.float32)
    

    bm3d_denoised = bm3d(
        noisy_scaled,
        sigma_psd=sigma_rician * 255 * 0.9,
        stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING
    )
    
    bm3d_denoised = np.clip(bm3d_denoised / 255.0, 0, 1)
    print("Image denoised using BM3D (DIPY noise estimation)")
    return bm3d_denoised

    
def save_samples(noise_vals, noisy_input_test, denoised_images1, denoised_images2,pure_test, img_height=256, img_width=256):
    noise_prop, noise_std, noise_mean = noise_vals
    plt.style.use('classic')
    fig, axes = plt.subplots(1, 6)
    fig.set_size_inches(24, 5)
    axes[0].set_title('Original image')
    axes[1].set_title('Noisy image')
    axes[2].set_title('Res UNet denoised image')
    axes[3].set_title('CBAM UNet denoised image')
    axes[4].set_title('NL Means denoised image')
    axes[5].set_title('BM3D denoised image')
    pure_images = []
    noisy_images = []
    bm3d_images = []
    nl_images = []

    for i in range(0, len(noisy_input_test)):
        # Get the sample and the reconstruction
        noisy_image = noisy_input_test[i][:, :, 0]
        pure_image = pure_test[i][:, :, 0]
        denoised_image1 = denoised_images1[i][:, :, 0]
        denoised_image2 = denoised_images2[i][:, :, 0]
        print(noisy_image.shape)
        bm3d_denoised =  bm3d_denoise_rician_dipy(noisy_image)
        bm3d_denoised = np.clip(bm3d_denoised, 0, 1)
        nl_denoised = conv_denoiser.nlm_denoise_dipy(noisy_image)
        nl_denoised = nl_denoised.clip(0, 1)
        noisy_images.append(noisy_image)
        pure_images.append(pure_image)
        bm3d_images.append(bm3d_denoised)
        nl_images.append(nl_denoised)
        # Plot sample and reconstruciton
        axes[0].imshow(pure_image, pyplot.cm.gray)
        axes[0].set_xlabel("SSIM: {:.5f}".format(measure1.get_image_ssim(pure_image, pure_image,data_range=1.0)))
        axes[1].imshow(noisy_image, pyplot.cm.gray)
        axes[1].set_xlabel("SSIM: {:.5f} PSNR: {:.5f}".format(measure1.get_image_ssim(pure_image, noisy_image,data_range=1.0),PSNR(pure_image, noisy_image,max_val=1.0)))
        axes[2].imshow(denoised_image1, pyplot.cm.gray)
        axes[2].set_xlabel("SSIM: {:.5f} PSNR: {:.5f}".format(measure1.get_image_ssim(pure_image, denoised_image1,data_range=1.0),PSNR(pure_image, denoised_image1,max_val=1.0)))
        axes[3].imshow(denoised_image2, pyplot.cm.gray)
        axes[3].set_xlabel("SSIM: {:.5f} PSNR: {:.5f}".format(measure1.get_image_ssim(pure_image, denoised_image2,data_range=1.0),PSNR(pure_image, denoised_image2,max_val=1.0)))
        axes[4].imshow(nl_denoised, pyplot.cm.gray)
        axes[4].set_xlabel("SSIM: {:.5f} PSNR: {:.5f}".format(measure1.get_image_ssim(pure_image, nl_denoised,data_range=1.0),PSNR(pure_image, nl_denoised,max_val=1.0)))
        axes[5].imshow(bm3d_denoised, pyplot.cm.gray)
        axes[5].set_xlabel("SSIM: {:.5f} PSNR: {:.5f}".format(measure1.get_image_ssim(pure_image , bm3d_denoised,data_range=1.0),PSNR(pure_image , bm3d_denoised,max_val=1.0)))

        print("pure_image range:", np.min(pure_image), np.max(pure_image))
        print("noisy_image range:", np.min(noisy_image), np.max(noisy_image))
        print("res_denoised_image range:", np.min(denoised_image1), np.max(denoised_image1))
        print("bm3d_denoised range:", np.min(bm3d_denoised), np.max(bm3d_denoised))
        print("nl_denoised range:", np.min(nl_denoised), np.max(nl_denoised))
        print("unet_denoised_image range:", np.min(denoised_image2), np.max(denoised_image2))



        
        fig.suptitle(
            "Medical Image Denoiser\nNoise Proportion: {0} - Mean: {1} - Standard deviation: {2}".format(noise_prop,
                                                                                                         noise_mean,
                                                                                                         noise_std),
            fontsize=14, fontweight='bold')

        plt.savefig("results/img({1},{2},{3}) {0}.png".format(i, noise_prop, noise_mean, noise_std))
        


    n1 = measure1.get_set_ssim(np.array(pure_images), np.array(noisy_images), img_height, img_width)
    n2 = measure1.get_set_ssim(np.array(pure_images), denoised_images1, img_height, img_width)
    n3 = measure1.get_set_ssim(np.array(pure_images) , np.array(bm3d_images), img_height, img_width)
    n4 = measure1.get_set_ssim(np.array(pure_images) , np.array(nl_images), img_height, img_width)
    n5=  measure1.get_set_ssim(np.array(pure_images), denoised_images2, img_height, img_width)
    
    psnr1 = get_set_psnr(np.array(pure_images), np.array(noisy_images),data_range=1.0)
    psnr2 = get_set_psnr(np.array(pure_images), denoised_images1, data_range=1.0)
    psnr3 = get_set_psnr(np.array(pure_images) , np.array(bm3d_images), data_range=1.0)
    psnr4 = get_set_psnr(np.array(pure_images) , np.array(nl_images), data_range=1.0)
    psnr5=  get_set_psnr(np.array(pure_images), denoised_images2, data_range=1.0)
    
    f = open("results/SSIM({0},{1},{2}) Results.txt".format(noise_prop, noise_mean, noise_std), "w")
    f.write("Noise Proportion: {0} - Mean: {1} - Standard Deviation: {2}\n".format(noise_prop, noise_mean, noise_std))
    f.write("Noisy SSIM:" + str(n1) + "\n")
    f.write("Res Unet Denoised SSIM:" + str(n2) + "\n")
    f.write("BM3D SSIM:" + str(n3) + "\n")
    f.write("NL Means SSIM:" + str(n4) + "\n")
    f.write("CBAM Unet Denoised SSIM:"+ str(n5) + "\n")
    f.close()

    print("Noisy SSIM: {0}, PSNR:{1}".format(n1,psnr1))
    print("Res Unet Denoised SSIM: {0}, PSNR:{1}".format(n2,psnr2))
    print("BM3D SSIM: {0}, PSNR:{1}".format(n3,psnr3))
    print("NL Means SSIM: {0}, PSNR:{1}".format(n4,psnr4))
    print("CBAM Unet Denoised SSIM: {0}, PSNR:{1}".format(n5,psnr5))
    


model_noskip = get_autoencoder_model_noskipnoCBAM()
model_Unet = get_autoencoder_model_NOCBAM()
model_CBAMunet =get_autoencoder_model_upsample()
#model_CBAMunet =get_autoencoder_model_transpose()


model_noskip.load_weights('./models/UNetNOSKIP.h5')
model_Unet.load_weights('./models/UNetNOCBAM.h5')
model_CBAMunet.load_weights('./models/UPsampleRicianBlurCBAM.h5')

    
    
denoised_images_noskip = model_noskip.predict(samples)

denoised_images_unet = model_Unet.predict(samples)

denoised_images_CBAMunet = model_CBAMunet.predict(samples)

print('input shape:',samples.shape)


print('plotting')
#save_samples((noise_prop, noise_std, noise_mean), samples, denoised_images_res,denoised_images_unet, pure_test,img_width=img_width, img_height=img_height)

save_samples_Ablation(samples, denoised_images_noskip, denoised_images_unet,denoised_images_CBAMunet,pure_test)

