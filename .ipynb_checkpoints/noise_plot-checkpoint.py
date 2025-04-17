import numpy as np
import matplotlib.pyplot as plt
from skimage import data, util
from scipy.ndimage import affine_transform
import glob
import os
from sklearn.model_selection import train_test_split
from dataset_reader import add_gaussian_noise,add_rician_noise
import cv2
from matplotlib import pyplot
from keras.models import load_model
import argparse
from sklearn.model_selection import train_test_split
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
import time
from tensorflow import image as tfimage


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
file_list = sorted(glob.glob(os.path.join("./data/extract/", "*.npz")))

#file_list = sorted(glob.glob(os.path.join("./data/crop/", "*.nii.gz")))

train_files, val_files = train_test_split(file_list, test_size=0.2, random_state=42)

val_files, test_files =  train_test_split(val_files, test_size=0.5, random_state=42)

batch_size = 10
steps_per_epoch = len(train_files) // batch_size
validation_steps = len(val_files) // batch_size

test_generator = read_ixi_t1_z(file_list=test_files, batch_size=10)

number_of_samples = 10
noisy_batch, clean_batch = next(test_generator)


samples = noisy_batch[:number_of_samples]
pure_test = clean_batch[:number_of_samples]


        
def save_samples(pure_test, img_height=256, img_width=256):
    plt.style.use('classic')
    fig, axes = plt.subplots(2, 3, figsize=(25, 15))
    fig.subplots_adjust(wspace=0.15, hspace=0.25,
                      left=0.03, right=0.97,
                      top=0.92, bottom=0.08)
    
    axes[0,0].set_title('Original image')
    axes[0,1].set_title('σ = 0.01')
    axes[0,2].set_title('σ = 0.03')
    axes[1,0].set_title('σ = 0.05') 
    axes[1,1].set_title('σ = 0.07')
    axes[1,2].set_title('σ = 0.09')

    for i in range(0, len(pure_test)):

        pure_image = pure_test[i][:, :, 0]
        blur = add_gaussian_blur(pure_image)
        noisy_image1 = add_rician_noise(blur,sigma=0.01)
        noisy_image2 = add_rician_noise(blur,sigma=0.03)
        noisy_image3 = add_rician_noise(blur,sigma=0.05)
        noisy_image4 = add_rician_noise(blur,sigma=0.07)
        noisy_image5 = add_rician_noise(blur,sigma=0.09)


        axes[0,0].imshow(pure_image, pyplot.cm.gray)
        axes[0,0].set_xlabel("SSIM: {:.5f}".format(measure1.get_image_ssim(pure_image, pure_image,data_range=1.0)))
        axes[0,1].imshow(noisy_image1, pyplot.cm.gray)
        axes[0,1].set_xlabel("SSIM: {:.5f} PSNR: {:.5f}".format(measure1.get_image_ssim(pure_image, noisy_image1,data_range=1.0),PSNR(pure_image, noisy_image1,max_val=1.0)))
        axes[0,2].imshow(noisy_image2, pyplot.cm.gray)
        axes[0,2].set_xlabel("SSIM: {:.5f} PSNR: {:.5f}".format(measure1.get_image_ssim(pure_image, noisy_image2,data_range=1.0),PSNR(pure_image, noisy_image2,max_val=1.0)))
        axes[1,0].imshow(noisy_image3, pyplot.cm.gray)
        axes[1,0].set_xlabel("SSIM: {:.5f} PSNR: {:.5f}".format(measure1.get_image_ssim(pure_image, noisy_image3,data_range=1.0),PSNR(pure_image, noisy_image3,max_val=1.0)))
        axes[1,1].imshow(noisy_image4, pyplot.cm.gray)
        axes[1,1].set_xlabel("SSIM: {:.5f} PSNR: {:.5f}".format(measure1.get_image_ssim(pure_image, noisy_image4,data_range=1.0),PSNR(pure_image, noisy_image4,max_val=1.0)))
        axes[1,2].imshow(noisy_image5, pyplot.cm.gray)
        axes[1,2].set_xlabel("SSIM: {:.5f} PSNR: {:.5f}".format(measure1.get_image_ssim(pure_image ,noisy_image5,data_range=1.0),PSNR(pure_image ,noisy_image5,max_val=1.0)))

        fig.suptitle(
            "Rician Noise In Different Standard deviation",
            fontsize=24, fontweight='bold')

        plt.savefig("results/img {0}.png".format(i))
        
    plt.close()
        
        
        
def save_samples_predict(pure_test,model, img_height=256, img_width=256):
    plt.style.use('classic')
    fig, axes = plt.subplots(2, 3)
    fig.set_size_inches(24, 16)
    axes[0,0].set_title('Original image')
    axes[0,1].set_title('σ = 0.01')
    axes[0,2].set_title('σ = 0.03')
    axes[1,0].set_title('σ = 0.05')
    axes[1,1].set_title('σ = 0.07')
    axes[1,2].set_title('σ = 0.09')


    for i in range(0, len(pure_test)):
        # Get the sample and the reconstruction
        pure_image = pure_test[i][:, :, 0]
        blur = add_gaussian_blur(pure_image)
        blur= np.expand_dims(np.expand_dims(blur, axis=0), axis=-1)
        noisy_image1 = add_rician_noise(blur,sigma=0.01)
        noisy_image1 = model.predict(noisy_image1)
        noisy_image1 = noisy_image1[0, :, :, 0]
        
        noisy_image2 = add_rician_noise(blur,sigma=0.03)
        noisy_image2 = model.predict(noisy_image2)
        noisy_image2 = noisy_image2[0, :, :, 0]
        
        noisy_image3 = add_rician_noise(blur,sigma=0.05)
        noisy_image3 = model.predict(noisy_image3)
        noisy_image3 = noisy_image3[0, :, :, 0]
        
        noisy_image4 = add_rician_noise(blur,sigma=0.07)
        noisy_image4 = model.predict(noisy_image4)
        noisy_image4 = noisy_image4[0, :, :, 0]
        
        noisy_image5 = add_rician_noise(blur,sigma=0.09)
        noisy_image5 = model.predict(noisy_image5)
        noisy_image5 = noisy_image5[0, :, :, 0]
        
        
        
        # Plot sample and reconstruciton
        axes[0,0].imshow(pure_image, pyplot.cm.gray)
        axes[0,0].set_xlabel("SSIM: {:.5f}".format(measure1.get_image_ssim(pure_image, pure_image,data_range=1.0)))
        axes[0,1].imshow(noisy_image1, pyplot.cm.gray)
        axes[0,1].set_xlabel("SSIM: {:.5f} PSNR: {:.5f}".format(measure1.get_image_ssim(pure_image, noisy_image1,data_range=1.0),PSNR(pure_image, noisy_image1,max_val=1.0)))
        axes[0,2].imshow(noisy_image2, pyplot.cm.gray)
        axes[0,2].set_xlabel("SSIM: {:.5f} PSNR: {:.5f}".format(measure1.get_image_ssim(pure_image, noisy_image2,data_range=1.0),PSNR(pure_image, noisy_image2,max_val=1.0)))
        axes[1,0].imshow(noisy_image3, pyplot.cm.gray)
        axes[1,0].set_xlabel("SSIM: {:.5f} PSNR: {:.5f}".format(measure1.get_image_ssim(pure_image, noisy_image3,data_range=1.0),PSNR(pure_image, noisy_image3,max_val=1.0)))
        axes[1,1].imshow(noisy_image4, pyplot.cm.gray)
        axes[1,1].set_xlabel("SSIM: {:.5f} PSNR: {:.5f}".format(measure1.get_image_ssim(pure_image, noisy_image4,data_range=1.0),PSNR(pure_image, noisy_image4,max_val=1.0)))
        axes[1,2].imshow(noisy_image5, pyplot.cm.gray)
        axes[1,2].set_xlabel("SSIM: {:.5f} PSNR: {:.5f}".format(measure1.get_image_ssim(pure_image ,noisy_image5,data_range=1.0),PSNR(pure_image ,noisy_image5,max_val=1.0)))
        
        fig.suptitle(
            "Model Denoising Performance Of Rician Noise In Different Standard deviation",
            fontsize=24, fontweight='bold')

        plt.savefig("results/img {0}.png".format(i))
        

model_SKIPCBAM = get_autoencoder_model_CBAMSKIP()
model_SKIPCBAM.load_weights('./models/CBAMINALLSKIPCONNECTION.h5')
model_CBAMunet =get_autoencoder_model_upsample()
model_CBAMunet.load_weights('./models/UPsampleRicianBlurCBAM.h5')

save_samples_predict(pure_test,model_CBAMunet,img_height=256, img_width=256)