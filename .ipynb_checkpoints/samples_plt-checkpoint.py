import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot

import conv_denoiser
import measure1


def plot_samples(noise_vals, noisy_input_test, denoised_images, pure_test, nu_samples=4, img_height=256, img_width=256):
    noise_prop, noise_std, noise_mean = noise_vals
    plt.style.use('classic')
    fig, axes = plt.subplots(1, 5)
    fig.set_size_inches(20, 5)
    axes[0].set_title('Original image')
    axes[1].set_title('Noisy image')
    axes[2].set_title('Autoencoder denoised image')
    axes[3].set_title('BM3D denoised image')
    axes[4].set_title('NL Means denoised image')
    pure_images = []
    noisy_images = []
    bm3d_images = []
    nl_images = []
    for i in range(0, len(noisy_input_test)):
        # Get the sample and the reconstruction
        noisy_image = noisy_input_test[i][:, :, 0]
        pure_image = pure_test[i][:, :, 0]
        denoised_image = denoised_images[i][:, :, 0]
        bm3d_denoised = conv_denoiser.bm3d_denoise(noisy_input_test[i].reshape(img_height, img_width))[0]
        bm3d_denoised = np.clip(bm3d_denoised, 0.0, 255.0)
        nl_denoised = conv_denoiser.nlm_denoise(noisy_input_test[i].reshape(img_height, img_width))[0]
        noisy_images.append(noisy_image)
        pure_images.append(pure_image)
        bm3d_images.append(bm3d_denoised)
        nl_images.append(nl_denoised)
        # Plot sample and reconstruciton
        axes[0].imshow(pure_image, pyplot.cm.gray)
        axes[0].set_xlabel("SSIM: {:.5f}".format(measure1.get_image_ssim(pure_image, pure_image,data_range=1.0)))
        axes[1].imshow(noisy_image, pyplot.cm.gray)
        axes[1].set_xlabel("SSIM: {:.5f}".format(measure1.get_image_ssim(pure_image, noisy_image,data_range=1.0)))
        axes[2].imshow(denoised_image, pyplot.cm.gray)
        axes[2].set_xlabel("SSIM: {:.5f}".format(measure1.get_image_ssim(pure_image, denoised_image,data_range=1.0)))
        axes[3].imshow(bm3d_denoised, pyplot.cm.gray)
        axes[3].set_xlabel("SSIM: {:.5f}".format(measure1.get_image_ssim(pure_image * 255.0, bm3d_denoised,data_range=255.0)))
        axes[4].imshow(nl_denoised, pyplot.cm.gray)
        axes[4].set_xlabel("SSIM: {:.5f}".format(measure1.get_image_ssim(pure_image * 255.0, nl_denoised,data_range=255.0)))

        print("pure_image range:", np.min(pure_image), np.max(pure_image))
        print("noisy_image range:", np.min(noisy_image), np.max(noisy_image))
        print("denoised_image range:", np.min(denoised_image), np.max(denoised_image))
        print("bm3d_denoised range:", np.min(bm3d_denoised), np.max(bm3d_denoised))
        print("nl_denoised range:", np.min(nl_denoised), np.max(nl_denoised))


    # Measure SSIM values for sampled images
    n1 = measure1.get_set_ssim(np.array(pure_images), np.array(noisy_images), img_height, img_width)
    n2 = measure1.get_set_ssim(np.array(pure_images) , denoised_images, img_height, img_width)
    n3 = measure1.get_set_ssim(np.array(pure_images) * 255.0, np.array(bm3d_images), img_height, img_width)
    n4 = measure1.get_set_ssim(np.array(pure_images) * 255.0, np.array(nl_images), img_height, img_width)
    print("Noisy SSIM: {0}".format(n1))
    print("Denoised SSIM: {0}".format(n2))
    print("BM3D SSIM: {0}".format(n3))
    print("NL Means SSIM: {0}".format(n4))
    fig.suptitle(
        "Medical Image Denoiser\nNoise Proportion: {0} - Mean: {1} - Standard Deviation: {2}\nSSIM Results-> Noisy: {3} - "
        "Denoised: {4} - BM3D: {5} - NL Means: {6}".format(noise_prop, noise_mean, noise_std, n1, n2, n3, n4),
        fontsize=14,
        fontweight='bold')
    # plt.savefig("output.png")
    plt.show()


def save_samples(noise_vals, noisy_input_test, denoised_images, pure_test, img_height=256, img_width=256):
    noise_prop, noise_std, noise_mean = noise_vals
    plt.style.use('classic')
    fig, axes = plt.subplots(1, 5)
    fig.set_size_inches(20, 5)
    axes[0].set_title('Original image')
    axes[1].set_title('Noisy image')
    axes[2].set_title('Autoencoder denoised image')
    axes[3].set_title('BM3D denoised image')
    axes[4].set_title('NL Means denoised image')
    pure_images = []
    noisy_images = []
    bm3d_images = []
    nl_images = []
    for i in range(0, len(noisy_input_test)):
        # Get the sample and the reconstruction
        noisy_image = noisy_input_test[i][:, :, 0]
        pure_image = pure_test[i][:, :, 0]
        denoised_image = denoised_images[i][:, :, 0]
        bm3d_denoised = conv_denoiser.bm3d_denoise(noisy_input_test[i].reshape(img_height, img_width))[0]
        bm3d_denoised = np.clip(bm3d_denoised, 0.0, 255.0)
        nl_denoised = conv_denoiser.nlm_denoise(noisy_input_test[i].reshape(img_height, img_width))[0]
        noisy_images.append(noisy_image)
        pure_images.append(pure_image)
        bm3d_images.append(bm3d_denoised)
        nl_images.append(nl_denoised)
        # Plot sample and reconstruciton
        axes[0].imshow(pure_image, pyplot.cm.gray)
        axes[0].set_xlabel("SSIM: {:.5f}".format(measure1.get_image_ssim(pure_image, pure_image,data_range=1.0)))
        axes[1].imshow(noisy_image, pyplot.cm.gray)
        axes[1].set_xlabel("SSIM: {:.5f}".format(measure1.get_image_ssim(pure_image, noisy_image,data_range=1.0)))
        axes[2].imshow(denoised_image, pyplot.cm.gray)
        axes[2].set_xlabel("SSIM: {:.5f}".format(measure1.get_image_ssim(pure_image, denoised_image,data_range=1.0)))
        axes[3].imshow(bm3d_denoised, pyplot.cm.gray)
        axes[3].set_xlabel("SSIM: {:.5f}".format(measure1.get_image_ssim(pure_image * 255.0, bm3d_denoised,data_range=255.0)))
        axes[4].imshow(nl_denoised, pyplot.cm.gray)
        axes[4].set_xlabel("SSIM: {:.5f}".format(measure1.get_image_ssim(pure_image * 255.0, nl_denoised,data_range=255.0)))

        print("pure_image range:", np.min(pure_image), np.max(pure_image))
        print("noisy_image range:", np.min(noisy_image), np.max(noisy_image))
        print("denoised_image range:", np.min(denoised_image), np.max(denoised_image))
        print("bm3d_denoised range:", np.min(bm3d_denoised), np.max(bm3d_denoised))
        print("nl_denoised range:", np.min(nl_denoised), np.max(nl_denoised))

        
        fig.suptitle(
            "Medical Image Denoiser\nNoise Proportion: {0} - Mean: {1} - Standard deviation: {2}".format(noise_prop,
                                                                                                         noise_mean,
                                                                                                         noise_std),
            fontsize=14, fontweight='bold')

        plt.savefig("results/img({1},{2},{3}) {0}.png".format(i, noise_prop, noise_mean, noise_std))

    n1 = measure1.get_set_ssim(np.array(pure_images), np.array(noisy_images), img_height, img_width)
    n2 = measure1.get_set_ssim(np.array(pure_images), denoised_images, img_height, img_width)
    n3 = measure1.get_set_ssim(np.array(pure_images) * 255.0, np.array(bm3d_images), img_height, img_width)
    n4 = measure1.get_set_ssim(np.array(pure_images) * 255.0, np.array(nl_images), img_height, img_width)

    f = open("results/SSIM({0},{1},{2}) Results.txt".format(noise_prop, noise_mean, noise_std), "w")
    f.write("Noise Proportion: {0} - Mean: {1} - Standard Deviation: {2}\n".format(noise_prop, noise_mean, noise_std))
    f.write("Noisy SSIM:" + str(n1) + "\n")
    f.write("Denoised SSIM:" + str(n2) + "\n")
    f.write("BM3D SSIM:" + str(n3) + "\n")
    f.write("NL Means SSIM:" + str(n4) + "\n")
    f.close()

    print("Noisy SSIM: {0}".format(n1))
    print("Denoised SSIM: {0}".format(n2))
    print("BM3D SSIM: {0}".format(n3))
    print("NL Means SSIM: {0}".format(n4))
