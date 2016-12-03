import numpy as np
from scipy.ndimage import imread
import matplotlib.pyplot as plt

from tvdd import striped_tvdd_2d


def main():
    image = imread("Lena512.png")
    # Add some noise
    isotropic_denoised = noisy_image = image + np.random.random(image.shape) * 0.09 * np.max(image)
    mu = 100.0

    isotropic_denoised = striped_tvdd_2d(isotropic_denoised, mu, order=2)

    plt.subplot(131)
    plt.imshow(image, cmap='gray')
    plt.subplot(132)
    plt.imshow(noisy_image, cmap='gray')
    plt.subplot(133)
    plt.imshow(isotropic_denoised, cmap='gray')
    plt.show()

if __name__ == '__main__':
    main()
