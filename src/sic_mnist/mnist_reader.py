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


MNIST_IMG_H = 28
MNIST_IMG_W = 28
MNIST_FEATURE_DIM = MNIST_IMG_H * MNIST_IMG_W

PATH_TRAIN_IMG = "../../dataset/raw/train-images-idx3-ubyte.gz"
PATH_TRAIN_LABEL = "../../dataset/raw/train-labels-idx1-ubyte.gz"
PATH_TEST_IMG = "../../dataset/raw/t10k-images-idx3-ubyte.gz"
PATH_TEST_LABEL = "../../dataset/raw/t10k-labels-idx1-ubyte.gz"

def read_data_sets_reshape():
    X_train = load_images(PATH_TRAIN_IMG).reshape(-1, MNIST_FEATURE_DIM)
    y_train = load_labels(PATH_TRAIN_LABEL)
    X_test = load_images(PATH_TEST_IMG).reshape(-1, MNIST_FEATURE_DIM)
    y_test = load_labels(PATH_TEST_LABEL)
    return X_train, y_train, X_test, y_test

if __name__ == "__main__":


    X_train, y_train, X_test, y_test = read_data_sets_reshape()
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    # 我想在这个地方就停止或者返回，不继续执行了
    raise SystemExit

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