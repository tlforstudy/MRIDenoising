import os
import re
import nibabel as nib
import cv2
import numpy as np
import glob
import ants


def read_pgm(filename, byteorder='>'):
    """Return image data from a raw PGM file as numpy array.

    Format specification: http://netpbm.sourceforge.net/doc/pgm.html

    """
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return np.frombuffer(buffer,
                            dtype='u1' if int(maxval) < 256 else byteorder + 'u2',
                            count=int(width) * int(height),
                            offset=len(header)
                            ).reshape((int(height), int(width)))


def read_ixi_t2(folder="data/ixi-T2/"):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)

        if img is not None:
            # img = cv2.resize(img, (64, 64))
            images.append(img)
    return np.array(images)


def read_covid(folder="data/covid/"):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)

        if img is not None:
            # img = cv2.resize(img, (64, 64))
            images.append(img)
    return np.array(images)

def add_gaussian_noise(pure, noise_mean=0.0, noise_std=1, noise_prop=0.1):
    noise = np.random.normal(noise_mean, noise_std, pure.shape)
    noisy_input = pure + noise_prop * noise
    noisy_input = np.clip(noisy_input,0,1)
    return noisy_input

def add_gaussian_blur(pure, sigma=0.8,kernel_size = 5):
    
    blurred = cv2.GaussianBlur(pure, (kernel_size, kernel_size), sigma)
    
    return blurred



def add_rician_noise(image, sigma=0.05):
    

    n1 = np.random.normal(0, sigma, image.shape)
    n2 = np.random.normal(0, sigma, image.shape)
    

    noisy = np.sqrt((image + n1)**2 + n2**2)
    

    noisy = np.clip(noisy, 0.0, 1.0)
    
    return noisy

def read_ixi_t1(file_list,batch_size=10,slices_per_file=3):
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

def read_mini_mias():
    images_tensor = np.zeros((322, 1024, 1024))
    i = 0
    for dirName, subdirList, fileList in os.walk("data/all-mias/"):
        for fname in fileList:
            if fname.endswith(".pgm"):
                images_tensor[i] = read_pgm("data/all-mias/" + fname, byteorder='<')
                i += 1
    return images_tensor


def read_dataset(path=None, img_width=64, img_height=64):
    try:
        images = []
        for filename in os.listdir(path):
            img = cv2.imread(os.path.join(path, filename), cv2.IMREAD_GRAYSCALE)

            if img is not None:
                img = cv2.resize(img, (img_width, img_height))
                images.append(img)
        return np.array(images)
    except:
        print("Error has occured during data loading")

        

def read_all_datasets():
    x = read_mini_mias()
    y = read_ixi_t2()
    z = read_covid()
    return x, y, z


if __name__ == "__main__":
    from matplotlib import pyplot

    x = read_ixi_t1()
    # image = read_pgm("data/all-mias/mdb001.pgm", byteorder='<')
    # images = numpy.zeros((322, 64, 64))
    # for i in range(x.shape[0]):
    #     images[i] = cv2.resize(x[i].reshape(1024, 1024), dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
    pyplot.imshow(x[0], pyplot.cm.gray)
    pyplot.show()
