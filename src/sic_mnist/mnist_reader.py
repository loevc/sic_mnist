import gzip
import struct
import os

import numpy as np


def load_images(path):
    print(os.path.abspath(path))
    with gzip.open(path, "rb") as f:
        magic, num_images, rows, cols = struct.unpack(
            ">IIII",
            f.read(16)
        )

        print("magic:", magic)
        print("num_images:", num_images)
        print("rows:", rows)
        print("cols:", cols)

        data = f.read()

    images = np.frombuffer(data, dtype=np.uint8)

    images = images.reshape(
        num_images,
        rows,
        cols
    )

    return images


def load_labels(path):
    print(os.path.abspath(path))
    with gzip.open(path, "rb") as f:
        magic, num_labels = struct.unpack(
            ">II",
            f.read(8)
        )

        print("magic:", magic)
        print("num_labels:", num_labels)

        data = f.read()

    labels = np.frombuffer(
        data,
        dtype=np.uint8
    )

    return labels


def read_data_sets():
    X_train = load_images("../../dataset/raw/train-images-idx3-ubyte.gz")
    y_train = load_labels("../../dataset/raw/train-labels-idx1-ubyte.gz")
    X_test = load_images("../../dataset/raw/t10k-images-idx3-ubyte.gz")
    y_test = load_labels("../../dataset/raw/t10k-labels-idx1-ubyte.gz")
    return X_train, y_train, X_test, y_test

if __name__ == "__main__":

    images = load_images(
        "../../dataset/raw/train-images-idx3-ubyte.gz"
    )

    labels = load_labels(
        "../../dataset/raw/train-labels-idx1-ubyte.gz"
    )

    print()
    # 这里是一个三维np数组
    print("images.shape =", images.shape)
    #
    print("labels.shape =", labels.shape)

    image = images[0]
    label = labels[0]

    print("image.shape =", image.shape)
    print("label =", label)

    print("min =", image.min())
    print("max =", image.max())

    # help(np.frombuffer)
    # print(np.frombuffer.__doc__)

    import matplotlib.pyplot as plt

    plt.imshow(image, cmap="gray")
    plt.title(f"label = {label}")
    plt.axis("off")
    plt.show()

    x = image.flatten()

    print(x.shape)

    print(image)
    print()
    print(image.flatten())