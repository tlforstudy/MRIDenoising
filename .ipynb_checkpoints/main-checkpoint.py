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

import matplotlib.pyplot as plt

def load_datasets(img_width=64, img_height=64):
    raw_mias = dataset_reader.read_mini_mias()  # Read mias dataset
    mias_images = np.zeros((raw_mias.shape[0], img_width, img_height))
    for i in range(raw_mias.shape[0]):
        mias_images[i] = cv2.resize(raw_mias[i], dsize=(img_width, img_height),
                                    interpolation=cv2.INTER_CUBIC)

    raw_t2 = dataset_reader.read_ixi_t2()  # Read ixi_t2 dataset
    t2_images = np.zeros((raw_t2.shape[0], img_width, img_width))
    for i in range(raw_t2.shape[0]):
        t2_images[i] = cv2.resize(raw_t2[i], dsize=(img_width, img_height),
                                       interpolation=cv2.INTER_CUBIC)
    
        #rawimages3 = dataset_reader.read_covid()  # Read covid dataset
        #images3 = np.zeros((329, img_width, img_width))
    #for i in range(rawimages3.shape[0]):
         #images3[i] = cv2.resize(rawimages3[i], dsize=(img_width, img_height),
                                #interpolation=cv2.INTER_CUBIC)
    return mias_images, t2_images  # , dental_images

'''
def add_noise(pure, pure_test):
    noise = np.random.normal(noise_mean, noise_std, pure.shape)  # np.random.poisson(1, pure.shape)
    noise_test = np.random.normal(noise_mean, noise_std, pure_test.shape)  # np.random.poisson(1, pure_test.shape)
    noisy_input = pure + noise_prop * noise
    noisy_input_test = pure_test + noise_prop * noise_test
    return noisy_input, noisy_input_test
'''

class Range(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __eq__(self, other):
        return self.start <= other <= self.end


if __name__ == "__main__":
    img_width, img_height = 256, 256
    batch_size = 10
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
    parser.add_argument("-tsplit", "--trainsplit", help="Train split [0-1] [default = 0.9]", type=float,
                        choices=[Range(0.0, 1.0)])
    parser.add_argument("-epoch", "--epoch", help="Number of epochs [default = 50]", type=int)
    parser.add_argument("-batch", "--batch", help="Batch size [default = 10]", type=int)
    parser.add_argument("-vsplit", "--validationsplit", help="Validation split [0-1] [default = 0.1]", type=float,
                        choices=[Range(0.0, 1.0)])
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
    if args.validationsplit:
        validation_split = args.validationsplit
    if args.trainsplit:
        train_split = args.trainsplit
    if args.shuffle:
        shuffle_test_set = True
    if args.size:
        img_width = args.size
        img_height = args.size

    print("[LOG] Loading datasets...")
    if args.load:
        print("[LOG] Loading data set from [{0}]".format(args.load))
        data_images = dataset_reader.read_dataset(args.load, img_width, img_height)
        input_train, input_test = CNN_denoiser.train_test_split1(data_images, train_split=train_split,
                                                                shuffle_test_set=shuffle_test_set, img_width=img_width,
                                                                img_height=img_height)  # Split 1 set to train and test sets
    else:
        print("Loading default datasets, MIAS and DX")
        '''mias_images, t2_images = load_datasets(img_width, img_height)  # Load mias and DX datasets
        train1, input_test = CNN_denoiser.train_test_split1(t2_images, train_split=train_split,
                                                                shuffle_test_set=shuffle_test_set, img_width=img_width,
                                                                img_height=img_height)  # Split 1 set to train and test sets
        input_train, test1 = CNN_denoiser.train_test_split1(t2_images, train_split=train_split,
                                                                shuffle_test_set=shuffle_test_set, img_width=img_width,
                                                                img_height=img_height)  # Split 1 set to train and test sets###
    print(
        "[LOG] Load completed\n" + "[LOG] Image size {0}x{1}".format(img_width,
                                                                    img_height) + "\n[LOG] Splitting datasets with [{0}] train set size\n[LOG] Shuffle test set: {1}".format(
            train_split, shuffle_test_set))
    #input_train, input_test = CNN_denoiser.train_test_split(dental_images,dx_images, train_split=train_split, shuffle_test_set=shuffle_test_set)  # Split 1 set to train and test sets
    # input_train, input_test = train_test_split3(mias_images, dx_images, dental_images, shuffle_test_set=True) '''
    
    
    file_list = glob.glob(os.path.join("./data/extract/", "*.npz"))
    train_files, val_files = train_test_split(file_list, test_size=0.2, random_state=42)
    val_files, test_files =  train_test_split(val_files, test_size=0.5, random_state=42)
    print(f"Total training files: {len(train_files)}")
    print(f"Total val files: {len(val_files)}")
    train_generator = dataset_reader.read_ixi_t1(train_files,batch_size=16)
    val_generator = dataset_reader.read_ixi_t1(val_files, batch_size=16)
    steps_per_epoch = len(train_files) // batch_size
    validation_steps = len(val_files) // batch_size

    
    
    #model_type=1 upsample, 2 transpose, 3 No CBAM, 4 No CBAM No skip
    model = CNN_denoiser(
    batch_size=16,
    nu_epochs=nu_epochs,
    validation_split=0.1,
    img_height=256,
    img_width=256,
    #model_type = 1
    #model_type = 2
    #model_type = 3
    #model_type = 4
    model_type = 5
)
    history = model.train_with_generator(
    train_generator=train_generator,
    val_generator=val_generator,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    save_path="./models/CBAMUNET.h5"
)
    
    
    #    cnn_denoiser.evaluate(noisy_input_test, pure_test)
    if args.plot:
        model.model_plots(noise_prop, noise_mean, noise_std)
    
    batch_size = min(10,len(val_files))

    test_generator = dataset_reader.read_ixi_t1(file_list=test_files, batch_size=batch_size)
    
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
    
    