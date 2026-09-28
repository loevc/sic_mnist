import numpy as np

from sic_mnist import mnist_reader


def softmax(z):
    z = z - np.max(z)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z)


def relu(x):
    return np.maximum(0, x)


def cross_entropy(p, y):
    return -np.sum(y * np.log(p + 1e-12))


def one_hot(label, num_classes=10):
    y = np.zeros(num_classes)
    y[label] = 1
    return y


input_size = 784
hidden_size = 128
output_size = 10

W1 = np.random.randn(hidden_size, input_size) * 0.01
b1 = np.zeros(hidden_size)

W2 = np.random.randn(output_size, hidden_size) * 0.01
b2 = np.zeros(output_size)


learning_rate = 0.01

X_train, y_train, X_test, y_test = mnist_reader.read_data_sets()

for i in range(1000):

    x = X_train[i].flatten()
    label = y_train[i]

    y = one_hot(label)

    # ==================
    # Forward
    # ==================

    z1 = W1 @ x + b1
    h = relu(z1)

    z2 = W2 @ h + b2
    p = softmax(z2)

    loss = cross_entropy(p, y)

    # ==================
    # Backward
    # ==================

    dz2 = p - y

    dW2 = np.outer(dz2, h)
    db2 = dz2

    dh = W2.T @ dz2

    dz1 = dh * (z1 > 0)

    dW1 = np.outer(dz1, x)
    db1 = dz1

    # ==================
    # Update
    # ==================

    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    if i % 100 == 0:
        print(
            f"step={i}, loss={loss:.4f}"
        )