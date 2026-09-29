import numpy as np
import mnist_reader


def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)

    exp_z = np.exp(z)

    return exp_z / np.sum(
        exp_z,
        axis=1,
        keepdims=True
    )



def one_hot_batch(labels, num_classes=10):

    Y = np.zeros(
        (len(labels), num_classes)
    )

    Y[np.arange(len(labels)), labels] = 1

    return Y


def cross_entropy_batch(P, Y):

    return -np.mean(
        np.sum(
            Y * np.log(P + 1e-12),
            axis=1
        )
    )

def relu(x):
    return np.maximum(0, x)


input_size = 784
hidden_size = 128
output_size = 10

W1 = np.random.randn(hidden_size, input_size) * 0.01
b1 = np.zeros(hidden_size)

W2 = np.random.randn(output_size, hidden_size) * 0.01
b2 = np.zeros(output_size)

batch_size = 32
learning_rate = 0.1

epochs = 10

X_train, y_train, X_test, y_test = mnist_reader.read_data_sets_reshape()

num_samples = len(X_train)

for epoch in range(epochs):

    # =========================
    # Shuffle
    # =========================

    indices = np.random.permutation(num_samples)

    X_train_shuffled = X_train[indices]
    y_train_shuffled = y_train[indices]

    epoch_loss = 0.0

    # =========================
    # Mini Batch
    # =========================

    for start in range(
        0,
        num_samples,
        batch_size
    ):

        end = start + batch_size

        X_batch = X_train_shuffled[start:end]
        labels = y_train_shuffled[start:end]

        current_batch_size = len(X_batch)

        Y = one_hot_batch(labels)

        # =========================
        # Forward
        # =========================

        Z1 = X_batch @ W1.T + b1
        H = relu(Z1)

        Z2 = H @ W2.T + b2
        P = softmax(Z2)

        loss = cross_entropy_batch(P, Y)

        epoch_loss += loss * current_batch_size

        # =========================
        # Backward
        # =========================

        dZ2 = P - Y

        dW2 = dZ2.T @ H / current_batch_size
        db2 = np.mean(
            dZ2,
            axis=0
        )

        dH = dZ2 @ W2

        dZ1 = dH * (Z1 > 0)

        dW1 = dZ1.T @ X_batch / current_batch_size
        db1 = np.mean(
            dZ1,
            axis=0
        )

        # =========================
        # Update
        # =========================

        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1

    epoch_loss /= num_samples

    print(
        f"epoch={epoch + 1}, "
        f"loss={epoch_loss:.4f}"
    )