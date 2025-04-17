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
from conv_denoiser import *
from measure1 import *
from skimage.transform import resize
import nibabel as nib
from dataset_reader import add_gaussian_noise,add_rician_noise
from skimage.draw import polygon
from skimage.metrics import structural_similarity as ssim
import ants
from scipy import ndimage as ndi
from tensorflow import image as tfimage
import tensorflow as tf
import bm3d
from dipy.denoise.noise_estimate import estimate_sigma as dipy_estimate_sigma
from autoencoder import get_autoencoder_model_upsample,get_autoencoder_model_transpose
from skimage.restoration import estimate_sigma
from skimage.util import img_as_float32
import time


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



def add_gaussian_blur(pure, sigma=0.8,kernel_size = 5):
    blurred = cv2.GaussianBlur(pure, (kernel_size, kernel_size), sigma)
    
    return blurred

            
def read_ixi_t1_z(file_list,batch_size=10,slices_per_file=5):
    num = len(file_list)
    while True:
        file_list = np.random.permutation(file_list)
        for i in range(0, num, batch_size):   
            batchfile = file_list[i:i+batch_size]
            noisy_batch, clean_batch,batch_128=[] , [],[]
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
                        noisy_128 = ndi.zoom(clean_slice, zoom=(0.5, 0.5))
                        #noisy_slice = add_gaussian_blur(clean_slice.copy())
                        #noisy_slice = add_gaussian_noise(noisy_slice, noise_mean=noise_mean, noise_std=noise_std, noise_prop=noise_prop)
                        noisy_slice = add_rician_noise(noisy_slice,sigma=0.05)
                        noisy_128 = add_rician_noise(noisy_128,sigma=0.05)
                        noisy_slice = np.expand_dims(noisy_slice, axis=-1)  # (H, W, 1)
                        clean_slice = np.expand_dims(clean_slice, axis=-1)
                        noisy_128 = np.expand_dims(noisy_128, axis=-1)
                        noisy_slice = np.clip(noisy_slice,0,1)
                        clean_slice = np.clip(clean_slice,0,1)
                        noisy_128 = np.clip(noisy_128,0,1)
                        
                        noisy_batch.append(noisy_slice)
                        clean_batch.append(clean_slice)
                        batch_128.append(noisy_128)
                
                
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
            
            yield np.array(noisy_batch), np.array(clean_batch),np.array(batch_128)

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



file_list = sorted(glob.glob(os.path.join("./data/extract/", "*.npz")))

#file_list = sorted(glob.glob(os.path.join("./data/crop/", "*.nii.gz")))

train_files, val_files = train_test_split(file_list, test_size=0.2, random_state=42)

val_files, test_files =  train_test_split(val_files, test_size=0.5, random_state=42)

batch_size = 10
steps_per_epoch = len(train_files) // batch_size
validation_steps = len(val_files) // batch_size

test_generator = read_ixi_t1_z(file_list=test_files, batch_size=10)

noisy_batch, clean_batch,batch_128 = next(test_generator)

samples = noisy_batch[:number_of_samples]
pure_test = clean_batch[:number_of_samples]
samples_128 = batch_128[:number_of_samples]




def evaluate_on_test_set(model,test_files,rician_noise_sigma=0.05,blur_sigma=0.8,kernel_size=5,model_name=None,img_height=256, img_width=256, batch_size=10):

    all_noisy = []
    all_clean = []
    
    for npz_file in test_files:
        try:
            with np.load(npz_file) as data:
                volume = data['data']
                valid_slices = data['valid_slices']
                

                for slice_idx in valid_slices:

                    clean_slice = volume[:, :, slice_idx]
                    noisy_slice = clean_slice.copy()
                    noisy_slice = add_gaussian_blur(clean_slice.copy(),sigma=blur_sigma,kernel_size=kernel_size)
                    noisy_slice = add_rician_noise(noisy_slice,sigma = rician_noise_sigma)
                    

                    noisy_slice = np.expand_dims(noisy_slice, axis=-1)
                    clean_slice = np.expand_dims(clean_slice, axis=-1)
                    
                    all_noisy.append(noisy_slice)
                    all_clean.append(clean_slice)
        except Exception as e:
            print(f"Error loading {npz_file}: {str(e)}")
            continue
    

    X_test = np.array(all_noisy)
    y_test = np.array(all_clean)
    start_time = time.time()
    denoised = model.predict(X_test)
    inference_time = time.time() - start_time
    print('The number of slices:',len(X_test))
    print('Inference time:',inference_time)
    ssim_values = []
    psnr_values = []
    
    ssim_noise_value=[]
    psnr_noise_values=[]
    for i in range(len(X_test)):
        pure = y_test[i,:,:,0]
        pred = denoised[i,:,:,0]
        noise =  X_test[i,:,:,0]
        
        
        ssim_test = get_set_ssim(pure, pred)
        psnr_test= get_set_psnr(pure, pred)

        ssim_noise = get_set_ssim(pure, noise)
        psnr_noise = get_set_psnr(pure,noise)
        
        ssim_values.append(ssim_test)
        psnr_values.append(psnr_test)
        ssim_noise_value.append(ssim_noise)
        psnr_noise_values.append(psnr_noise)
        
    avg_ssim = np.mean(ssim_values)
    avg_psnr = np.mean(psnr_values)
    
    avg_noise_ssim = np.mean(ssim_noise_value)
    avg_noise_psnr = np.mean(psnr_noise_values)
    
    print("Model performance on test set, SSIM: {0}, PSNR:{1}".format(avg_ssim,avg_psnr))
    print("Noisy image on test set, SSIM: {0}, PSNR:{1}".format(avg_noise_ssim,avg_noise_psnr))
    



'''  
for model, model_name in [(model, "CBAM in Encoder and bottleneck")]:
    print(f"\nEvaluating model: {model_name}")
    evaluate_on_test_set(model, test_files,model_name=model_name)
'''
       
        
        
        
        
def save_samples_perdict(pure_test,denoised, samples,upsampled,denoised_128_img,noisy_downsample,img_height=256, img_width=256):
    plt.style.use('classic')
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    fig.subplots_adjust(wspace=0.15, hspace=0.25,
                      left=0.03, right=0.97,
                      top=0.92, bottom=0.08)
    axes[0,0].set_title('Original image')
    axes[0,1].set_title('Noisy image')
    axes[1,0].set_title('Residual U-Net based on\n 256x256 image training')
    axes[1,1].set_title('Upsampling results for Residual U-Net\n based on 128x128 image training')





    for i in range(0, len(pure_test)):
        # Get the sample and the reconstruction
        pure_image = pure_test[i][:, :, 0]
        noisy_image = samples[i][:, :, 0]
        

        noisy_image1 = denoised[i][:, :, 0]
        
        noisy_image2 = upsampled[i][:, :, 0]
        

        
        
        # Plot sample and reconstruciton
        axes[0,0].imshow(pure_image, pyplot.cm.gray)
        axes[0,0].set_xlabel("SSIM: {:.5f}".format(get_image_ssim(pure_image, pure_image,data_range=1.0)))
        axes[0,1].imshow(noisy_image, pyplot.cm.gray)
        axes[0,1].set_xlabel("SSIM: {:.5f} PSNR: {:.5f}".format(get_image_ssim(pure_image, noisy_image,data_range=1.0),PSNR(pure_image, noisy_image,max_val=1.0)))
        axes[1,0].imshow(noisy_image1, pyplot.cm.gray)
        axes[1,0].set_xlabel("SSIM: {:.5f} PSNR: {:.5f}".format(get_image_ssim(pure_image, noisy_image1,data_range=1.0),PSNR(pure_image, noisy_image1,max_val=1.0)))
        axes[1,1].imshow(noisy_image2, pyplot.cm.gray)
        axes[1,1].set_xlabel("SSIM: {:.5f} PSNR: {:.5f}".format(get_image_ssim(pure_image, noisy_image2,data_range=1.0),PSNR(pure_image, noisy_image2,max_val=1.0)))


        fig.suptitle(
            "Comparison of noise removal performance of Residual U-Net on \n256x256 images based on two different image sizes training",
            fontsize=24, fontweight='bold')

        plt.savefig("results/img_resunet {0}.png".format(i))
    
    
model_res_256  = load_model('./models/res_skip_cae_Rician256.h5',custom_objects=custom_objects2)
model_res_128 = load_model('./models/res_skip_cae_denoiser.h5',custom_objects=custom_objects2)



res256img = model_res_256.predict(samples)
res128img = model_res_128.predict(samples_128)

upsampled = ndi.zoom(res128img, zoom=(1, 2, 2, 1), order = 1)


save_samples_perdict(pure_test,res256img,samples,upsampled,res128img,samples_128,img_height=256, img_width=256)
