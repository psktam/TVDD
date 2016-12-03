import numpy as np
from scipy.signal import medfilt
import matplotlib.pyplot as plt

from tvdd import tvdd_mm


def main():
    t = np.linspace(0, 10 * 2 * np.pi, 1000)
    noise = 0.1 * (np.random.random(len(t)) - 0.5)
    orig_sin = np.sin(t)
    noisy_sin = orig_sin + noise
    med_sin = medfilt(noisy_sin, 11)
    cleaned_sin = tvdd_mm(noisy_sin, 50.0, 2)

    orig_square = np.array((([0.0] * 100 + [1.0] * 100) * (len(t) / 19))[:len(t)])
    noisy_square = orig_square + noise
    med_square = medfilt(noisy_square, 11)
    cleaned_square = tvdd_mm(noisy_square, 0.5, 2)

    for idx, (orig, noisy, cleaned, med) in enumerate([(orig_sin, noisy_sin, cleaned_sin, med_sin),
                                                       (orig_square, noisy_square, cleaned_square, med_square)]):
        plt.subplot(int('12{}'.format(idx + 1)))
        plt.plot(t, noisy, label='noisy')
        plt.plot(t, cleaned, label='cleaned')
        plt.plot(t, med, label='median')
        plt.plot(t, orig, label='original')
        plt.legend(loc='best', fontsize='small', framealpha=0.5)
    plt.show()


if __name__ == '__main__':
    main()
