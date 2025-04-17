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



def add_gaussian_blur(pure, sigma=0.8):
    kernel_size = 5
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

#file_list = sorted(glob.glob(os.path.join("./data/crop/", "*.nii.gz")))

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


def bm3d_denoise_rician_dipy(noisy_image):

    noisy_image = img_as_float32(noisy_image)
    noisy_image = np.clip(noisy_image, 0, None)
    noisy_image = noisy_image[:,:,np.newaxis]
    sigma_rician = dipy_estimate_sigma(noisy_image, N=4)
    

    noisy_scaled = (noisy_image * 255.0).astype(np.float32)
    

    bm3d_denoised = bm3d.bm3d(
        noisy_scaled,
        sigma_psd=0.05*255,
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
    axes[2].set_title('UNet with upsample denoised image')
    axes[3].set_title('UNet with transpose denoised image')

    pure_images = []
    noisy_images = []

    for i in range(0, len(noisy_input_test)):
        # Get the sample and the reconstruction
        noisy_image = noisy_input_test[i][:, :, 0]
        pure_image = pure_test[i][:, :, 0]
        denoised_image1 = denoised_images1[i][:, :, 0]
        denoised_image2 = denoised_images2[i][:, :, 0]
        
        noisy_images.append(noisy_image)
        pure_images.append(pure_image)

        # Plot sample and reconstruciton
        axes[0].imshow(pure_image, pyplot.cm.gray)
        axes[0].set_xlabel("SSIM: {:.5f}".format(get_image_ssim(pure_image, pure_image,data_range=1.0)))
        axes[1].imshow(noisy_image, pyplot.cm.gray)
        axes[1].set_xlabel("SSIM: {:.5f} PSNR: {:.5f}".format(get_image_ssim(pure_image, noisy_image,data_range=1.0),PSNR(pure_image, noisy_image,max_val=1.0)))
        axes[2].imshow(denoised_image1, pyplot.cm.gray)
        axes[2].set_xlabel("SSIM: {:.5f} PSNR: {:.5f}".format(get_image_ssim(pure_image, denoised_image1,data_range=1.0),PSNR(pure_image, denoised_image1,max_val=1.0)))
        axes[3].imshow(denoised_image2, pyplot.cm.gray)
        axes[3].set_xlabel("SSIM: {:.5f} PSNR: {:.5f}".format(get_image_ssim(pure_image, denoised_image2,data_range=1.0),PSNR(pure_image, denoised_image2,max_val=1.0)))


        
        fig.suptitle(
            "Rician noise σ=0.05, Gaussian Blur σ=0.8 kernel=5",
            fontsize=20, fontweight='bold')

        plt.savefig("results/img({1},{2},{3}) {0}.png".format(i, noise_prop, noise_mean, noise_std))
    
    

    n1 = get_set_ssim(np.array(pure_images), np.array(noisy_images), img_height, img_width)
    n2 = get_set_ssim(np.array(pure_images), denoised_images1, img_height, img_width)
    n5=  get_set_ssim(np.array(pure_images), denoised_images2, img_height, img_width)
    
    psnr1 = get_set_psnr(np.array(pure_images), np.array(noisy_images),data_range=1.0)
    psnr2 = get_set_psnr(np.array(pure_images), denoised_images1, data_range=1.0)
    psnr5=  get_set_psnr(np.array(pure_images), denoised_images2, data_range=1.0)
    
    f = open("results/SSIM({0},{1},{2}) Results.txt".format(noise_prop, noise_mean, noise_std), "w")
    f.write("Noise Proportion: {0} - Mean: {1} - Standard Deviation: {2}\n".format(noise_prop, noise_mean, noise_std))
    f.write("Noisy SSIM:" + str(n1) + "\n")
    f.write("CBAM Unet with upsampling Denoised SSIM:" + str(n2) + "\n")
    f.write("CBAM Unet with transpose Denoised SSIM:"+ str(n5) + "\n")
    f.close()

    print("Noisy SSIM: {0}, PSNR:{1}".format(n1,psnr1))
    print("CBAM Unet with upsampling SSIM: {0}, PSNR:{1}".format(n2,psnr2))
    print("CBAM Unet with transpos SSIM: {0}, PSNR:{1}".format(n5,psnr5))
    


def get_cbam_activations(model, cbam_layer_name, input_image):


    cbam_layer_idx = None
    for i, layer in enumerate(model.layers):
        if layer.name == cbam_layer_name:
            cbam_layer_idx = i
            break
    
    if cbam_layer_idx is None:
        raise ValueError(f"Layer {cbam_layer_name} not found in model")

  
    input_model = tf.keras.Model(
        inputs=model.input,
        outputs=model.layers[cbam_layer_idx].input
    )
    
    
    output_model = tf.keras.Model(
        inputs=model.input,
        outputs=model.layers[cbam_layer_idx].output
    )

    
    cbam_input = input_model.predict(input_image[0:1])
    cbam_output = output_model.predict(input_image[0:1])
    return cbam_input, cbam_output 
    
    

def compute_spatial_weights(cbam_input, cbam_output):
    epsilon = 1e-7
    ratio = np.squeeze(cbam_output) / (np.squeeze(cbam_input) + epsilon)
    return np.mean(ratio, axis=-1)

def compute_channel_weights(cbam_input, cbam_output):
    epsilon = 1e-7
    input_mean = np.squeeze(np.mean(cbam_input, axis=(1, 2)))
    output_mean = np.squeeze(np.mean(cbam_output, axis=(1, 2)))
    return output_mean / (input_mean + epsilon)


def visualize_cbam_weights_without_modification(model,model_name ,test_img):
    cbam_layers = [layer for layer in model.layers if 'cbam' in layer.name.lower()]
    plt.figure(figsize=(15, 5 * len(cbam_layers)))
    
    for idx, cbam_layer in enumerate(cbam_layers):
        cbam_input, cbam_output = get_cbam_activations(model, cbam_layer.name, test_img)
        

        channel_weights = compute_channel_weights(cbam_input, cbam_output)
        spatial_weights = compute_spatial_weights(cbam_input, cbam_output)
        
        plt.subplot(len(cbam_layers), 2, 2*idx + 1)
        plt.bar(range(len(channel_weights)), channel_weights)
        plt.title(f"CBAM Block {idx+1} - Channel Attention")
        plt.xlabel("Channel Index")

        plt.subplot(len(cbam_layers), 2, 2*idx + 2)
        plt.imshow(spatial_weights, cmap='jet')
        plt.title(f"CBAM Block {idx+1} - Spatial Attention")
        plt.colorbar()
    
    plt.tight_layout()
    plt.savefig("./results/cbam_heatmap {0}.pdf".format(model_name), bbox_inches='tight', dpi=300)
    plt.close()

    
    

model_CBAMINALLENCODER = get_autoencoder_model_CBAMencoder()
model_CBAMINALLSKIPCONNECTION = get_autoencoder_model_CBAMSKIP()
model_CBAMunet =get_autoencoder_model_upsample()


model_CBAMINALLENCODER.load_weights('./models/CBAMINALLENCODER.h5')
model_CBAMINALLSKIPCONNECTION.load_weights('./models/CBAMINALLSKIPCONNECTION.h5')
model_CBAMunet.load_weights('./models/UPsampleRicianBlurCBAM.h5')





        
for model, model_name in [(model_CBAMunet, "Normal"),(model_CBAMINALLENCODER,"All_Encoder"),(model_CBAMINALLSKIPCONNECTION,"All_skip")]:
    print(f"\nVisualize model: {model_name}")
    visualize_cbam_weights_without_modification(model,model_name,samples)