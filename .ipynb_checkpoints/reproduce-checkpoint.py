import argparse
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import glob
import os
import dataset_reader
import samples_plt
from CNN_denoiser import CNN_denoiser
import tensorflow as tf
from dataset_reader import add_gaussian_noise,add_rician_noise
import ants
from tensorflow import math as tfmath
from scipy import ndimage as ndi
from tensorflow import image as tfimage
import json


__author__ = "Adrian Arnaiz-Rodriguez"
__email__ = "aarnaizr@uoc.edu"
__version__ = "1.0.0"


from tensorflow import Tensor
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, ReLU, BatchNormalization, Add, MaxPooling2D, UpSampling2D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.regularizers import l2

def relu_bn(inputs: Tensor, name='RB') -> Tensor:
    y = BatchNormalization(name=name+'inner_BN')(inputs) 
    return ReLU(name=name+'_innerReLu')(y)

def original_residual_block(x: Tensor, filters, ks = (3,3), stride = 2, name='RB',ker_reg=None):
    y = Conv2D(filters= filters,
               kernel_size= ks,
               strides= stride,              
               padding="same",
               kernel_regularizer=ker_reg,
               name=name+'_C1')(x)
    y = relu_bn(y, name=name)
    
    y = Conv2D(filters= filters,
               kernel_size= ks,
               strides= 1,               
               padding="same",
               kernel_regularizer=ker_reg,
               name=name+'_C2')(y)
    y = BatchNormalization(name=name+'_BN')(y) 
    
    if stride !=1:
        x = Conv2D(filters = filters,
                   kernel_size= (1,1),
                   strides= stride,
                   padding="same",
                   kernel_regularizer=ker_reg,
                   name=name+'_CAdjust')(x)
    
    y = Add(name=name+'_ResSUM')([x,y])
    y =  ReLU(name=name+'_ReLu')(y)
    return y

def full_pre_residual_block(x: Tensor, filters, ks = (3,3), stride = 2, name='FPRB', ker_reg=None):
    '''
    https://towardsdatascience.com/an-overview-of-resnet-and-its-variants-5281e2f56035
    [7]. K. He, X. Zhang, S. Ren, and J. Sun. Identity Mappings in Deep Residual Networks. arXiv preprint arXiv:1603.05027v3,2016.
    '''
    y = relu_bn(x, name=name+'_BN_R1')
    y = Conv2D(filters= filters,
               kernel_size= ks,
               strides= stride,              
               padding="same",
               kernel_regularizer= ker_reg,
               name=name+'_C1')(y)
     
    y = relu_bn(y, name=name+'_BN_R2')
    y = Conv2D(filters= filters,
               kernel_size= ks,
               strides= 1,               
               padding="same",
               kernel_regularizer=ker_reg,
               name=name+'_C2')(y)
    
    if stride !=1:
        x = Conv2D(filters = filters,
                   kernel_size= (1,1),
                   strides= stride,
                   padding="same",
                   kernel_regularizer=ker_reg,
                   name=name+'_CAdjust')(x) 
    
    y = Add(name=name+'_ResSUM')([x,y])
    return y


def upsampling_block(x: Tensor, filters, ks=(3,3), name='UP'):
    '''y = Conv2D(filters, ks, activation='relu', padding='same', name=name+'_C1')(x) #474k
    y = BatchNormalization(name=name+'_BN')(y) 
    y = UpSampling2D((2, 2), name=name+'_Up')(y)'''
    
    y = Conv2DTranspose(filters, ks, strides=(2,2), padding='same', name=name+'_C1')(x) #474k
    y = relu_bn(y, name=name+'_BN_RUP')
    return y
    

def build_res_skip_cae(input_shape, block_type='original', ker_reg = False):
    #INPUT
    input_img = Input(shape = input_shape) #128x128x1
    
    if block_type == 'original':
        residual_block = original_residual_block
        bname = 'RB'
    elif block_type == 'full_pre':
        residual_block = full_pre_residual_block
        bname = 'FP_RB'
    else:
        raise Exception('Not implemented block')

    ker_reg = l2(1e-5) if ker_reg else None

    #ENCODER
    x1 = Conv2D(32, (3,3), strides= 2, padding="same", name='Conv1', kernel_regularizer=ker_reg)(input_img) #64x64x32
    x = residual_block(x1, 64, name=bname+'1', ker_reg=ker_reg) #32x32x64
    x2 = residual_block(x, 64, stride=1, name=bname+'2_same_dim', ker_reg=ker_reg) #32x32x64
    latent = residual_block(x2, 128, stride=2, name=bname+'3', ker_reg=ker_reg) #16x16x128
    
    #DECODER
    y = upsampling_block(latent,64, name='UP1') #32*32*64
    y = Concatenate(name='SKIP_CONN1')([y, x2])
    y = upsampling_block(y,32, name='UP2') #64*64*32
    y = Concatenate(name='SKIP_CONN2')([y, x1])
    y = upsampling_block(y,16, name='UP3') #128*128*16
    decoded = Conv2D(1, (3,3), activation='sigmoid', padding='same', kernel_regularizer=ker_reg)(y)
    return Model(input_img, decoded)



def read_ixi_t1_z_128(file_list,batch_size=10,slices_per_file=3):
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
                        noisy_slice = add_gaussian_blur(clean_slice.copy())
                        #noisy_slice = add_gaussian_noise(noisy_slice)
                        noisy_slice = add_rician_noise(noisy_slice)
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

def add_gaussian_blur(pure, sigma=0.8):
    kernel_size = 5
    blurred = cv2.GaussianBlur(pure, (kernel_size, kernel_size), sigma)
    
    return blurred

if __name__ == "__main__":
    
    img_width, img_height = 256,256
    batch_size = 16
    nu_epochs = 10
    validation_split = 0.1
    train_split = 0.9
    verbosity = 1
    noise_prop = 0.1
    noise_std = 1
    noise_mean = 0
    number_of_samples = 4
    shuffle_test_set = False

    parser = argparse.ArgumentParser(description='Image Denoiser')
    parser.add_argument("-load", "--load", help="Path of dataset to load [default = DX and MIAS are loaded]", type=str)
    parser.add_argument("-size", "--size", help="Image size 64x64 or 128x128 [choices = 128, 64] [default = 64]",
                        type=int,
                        choices=[128, 64])
    parser.add_argument("-p", "--proportion", help="Gaussian noise proportion [default = 0.1]", type=float)
    parser.add_argument("-std", "--sdeviation", help="Gaussian noise standard deviation [default = 1]", type=float)
    parser.add_argument("-m", "--mean", help="Gaussian noise mean [default = 0]", type=float)
    parser.add_argument("-s", "--samples", help="Number of samples [default = 4]", type=int)
    parser.add_argument("-shuffle", "--shuffle", help="Shuffle test set", action="store_true")
    parser.add_argument("-epoch", "--epoch", help="Number of epochs [default = 50]", type=int)
    parser.add_argument("-batch", "--batch", help="Batch size [default = 10]", type=int)
    parser.add_argument("-save", "--save", help="Save test set samples", action="store_true")
    parser.add_argument("-plot", "--plot", help="Plot model loss", action="store_true")
    args = parser.parse_args()
    if args.proportion:
        noise_prop = args.proportion
    if args.sdeviation:
        noise_std = args.sdeviation
    if args.mean:
        noise_mean = args.mean
    if args.samples:
        number_of_samples = args.samples
    if args.epoch:
        nu_epochs = args.epoch
    if args.batch:
        batch_size = args.batch
    if args.shuffle:
        shuffle_test_set = True
    if args.size:
        img_width = args.size
        img_height = args.size

    def DSSIM(y_true, y_pred):
        return tfmath.divide(tfmath.subtract(1.0,tfimage.ssim(y_true, y_pred, max_val=1.0)),2.0)

        

    file_list = sorted(glob.glob(os.path.join("./data/extract/", "*.npz")))
    train_files, val_files = train_test_split(file_list, test_size=0.2, random_state=42)
    val_files, test_files =  train_test_split(val_files, test_size=0.5, random_state=42)
    print(f"Total training files: {len(train_files)}")
    print(f"Total val files: {len(val_files)}")
    train_generator = read_ixi_t1_z_128(train_files,batch_size=16)
    val_generator = read_ixi_t1_z_128(val_files, batch_size=16)
    steps_per_epoch = len(train_files) // batch_size
    validation_steps = len(val_files) // batch_size

    input_shape = (img_height, img_width, 1)
    model = build_res_skip_cae(
        input_shape=input_shape,
        block_type='original',
        ker_reg=True
    )
    model.compile(optimizer='adam', loss=DSSIM)


    history = model.fit(
        train_generator,
        epochs=nu_epochs,
        validation_data=val_generator,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps
    )

    model.save("./models/res_skip_cae_denoiser256.h5")
    with open("./results/training_history.json", "w") as f:
        json.dump(history.history, f)
    #    cnn_denoiser.evaluate(noisy_input_test, pure_test)
    if args.plot:
        model.model_plots(noise_prop, noise_mean, noise_std)
    
    batch_size = min(10,len(val_files))

    test_generator = read_ixi_t1_z_128(file_list=test_files, batch_size=batch_size)
    
    noisy_batch, clean_batch = next(test_generator)

    
    samples = noisy_batch[:number_of_samples]
    pure_test = clean_batch[:number_of_samples]
    
    
    print("[LOG] Training and model evaluation completed\n[LOG] Denoising images test set...")
    denoised_images = model.predict(samples)
    print("Output min:", np.min(denoised_images), "max:", np.max(denoised_images))

    print("[LOG] Image denoising completed\n[LOG] Plotting denoised samples")
    samples_plt.plot_samples((noise_prop, noise_std, noise_mean), samples, denoised_images, pure_test,
                             number_of_samples, img_width=img_width, img_height=img_height)
    if args.save:
        samples_plt.save_samples((noise_prop, noise_std, noise_mean), samples, denoised_images, pure_test,
                                 img_width=img_width, img_height=img_height)
    
    