import numpy as np
from scipy.ndimage import imread
import matplotlib.pyplot as plt

from tvdd import tvdd_sb_itv


def main():
    image = imread("Lena512.png")
    # Add some noise
    isotropic_denoised = noisy_image = image + np.random.random(image.shape) * 0.09 * np.max(image)
    mu = 25.0

    isotropic_denoised = tvdd_sb_itv(isotropic_denoised, mu)

    plt.subplot(131)
    plt.imshow(image, cmap='gray')
    plt.subplot(132)
    plt.imshow(noisy_image, cmap='gray')
    plt.subplot(133)
    plt.imshow(isotropic_denoised, cmap='gray')
    plt.show()

if __name__ == '__main__':
    main()
