import numpy as np
import mnist_reader
import time



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

W1 = np.random.randn(hidden_size, input_size) * np.sqrt(2.0 / input_size)
b1 = np.zeros(hidden_size)

W2 = np.random.randn(output_size, hidden_size) * np.sqrt(2.0 / input_size)
b2 = np.zeros(output_size)

batch_size = 32
learning_rate = 0.1
# learning_rate = 0.01

epochs = 5

X_train, y_train, X_test, y_test = mnist_reader.read_data_sets_reshape()

print("X_train dtype:", X_train.dtype)
print("X_train min:", X_train.min())
print("X_train max:", X_train.max())
print("X_train mean:", X_train.mean())
print("X_train std:", X_train.std())

X_train = X_train.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0

num_samples = len(X_train)


start_time = time.perf_counter()

loss_history = []

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

    loss_history.append(epoch_loss)

    print(
        f"epoch={epoch + 1}, "
        f"loss={epoch_loss:.4f}, "
        f"p={np.exp(-epoch_loss):.4f}"
    )

print(f"cost: {time.perf_counter() - start_time:.4f} s")


def predict(X):

    Z1 = X @ W1.T + b1
    H = relu(Z1)

    Z2 = H @ W2.T + b2

    P = softmax(Z2)

    return np.argmax(P, axis=1)


predictions = predict(X_test)

accuracy = np.mean(
    predictions == y_test
)

print(
    f"test accuracy: {accuracy:.4f}"
)


import matplotlib.pyplot as plt

plt.plot(loss_history)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")

plt.show()


def accuracy(X, y):

    predictions = predict(X)

    return np.mean(
        predictions == y
    )


train_accuracy = accuracy(
    X_train,
    y_train
)

test_accuracy = accuracy(
    X_test,
    y_test
)

print(
    f"train accuracy: {train_accuracy:.4f}"
)

print(
    f"test accuracy: {test_accuracy:.4f}"
)

predictions = predict(X_test)

wrong_indices = np.where(
    predictions != y_test
)[0]

print(
    "wrong samples:",
    len(wrong_indices)
)

index = wrong_indices[0]

image = X_test[index]

true_label = y_test[index]

predicted_label = predictions[index]


plt.imshow(
    image.reshape(28, 28),
    cmap="gray"
)

plt.title(
    f"True: {true_label}, "
    f"Predicted: {predicted_label}"
)

plt.axis("off")

plt.show()


plt.figure(figsize=(8, 8))

for i in range(16):

    index = wrong_indices[i]

    image = X_test[index]

    true_label = y_test[index]

    predicted_label = predictions[index]

    plt.subplot(4, 4, i + 1)

    plt.imshow(
        image.reshape(28, 28),
        cmap="gray"
    )

    plt.title(
        f"T:{true_label} P:{predicted_label}"
    )

    plt.axis("off")

plt.tight_layout()
plt.show()


def confusion_matrix(
    y_true,
    y_pred,
    num_classes=10
):

    matrix = np.zeros(
        (num_classes, num_classes),
        dtype=int
    )

    for true, pred in zip(
        y_true,
        y_pred
    ):

        matrix[true, pred] += 1

    return matrix



cm = confusion_matrix(
    y_test,
    predictions
)

print(cm)



plt.figure(figsize=(8, 8))

plt.imshow(cm)

plt.colorbar()

plt.xlabel("Predicted")

plt.ylabel("True")

plt.title("Confusion Matrix")

plt.xticks(range(10))

plt.yticks(range(10))

plt.show()