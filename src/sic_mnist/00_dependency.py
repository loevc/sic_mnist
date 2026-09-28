import numpy as np
import matplotlib.pyplot as plt


def main():
    print("NumPy:", np.__version__)
    print("Matplotlib:", plt.matplotlib.__version__)

    image = np.random.rand(28, 28)

    print("image shape:", image.shape)
    print("image ndim:", image.ndim)


if __name__ == "__main__":
    main()