import numpy as np
import mnist_reader

def softmax(z):
    z = z - np.max(z)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z)

def cross_entropy(p, y):
    return -np.sum(y * np.log(p + 1e-12))

def train_step(x, y, W, b, learning_rate):

    # ----------------
    # 1. Forward
    # ----------------
    z = W @ x + b
    p = softmax(z)

    # ----------------
    # 2. Loss
    # ----------------
    loss = cross_entropy(p, y)

    # ----------------
    # 3. Backward
    # ----------------
    dz = p - y

    dW = np.outer(dz, x)
    db = dz

    # ----------------
    # 4. Gradient Descent
    # ----------------
    W -= learning_rate * dW
    b -= learning_rate * db

    return loss

def one_hot(label, num_classes=10):
    y = np.zeros(num_classes)
    y[label] = 1
    return y

learning_rate = 0.01

W = np.random.randn(10, 784) * 0.01
b = np.zeros(10)

losses = []

X_train, y_train, X_test, y_test = mnist_reader.read_data_sets()

for i in range(1000):

    x = X_train[i].flatten()

    label = y_train[i]
    y = one_hot(label)

    loss = train_step(
        x,
        y,
        W,
        b,
        learning_rate
    )

    losses.append(loss)

    if i % 100 == 0:
        print(
            f"step={i}, loss={loss:.4f}"
        )

def predict(x, W, b):

    z = W @ x + b
    p = softmax(z)

    return np.argmax(p)

correct = 0

for i in range(1000):

    prediction = predict(
        X_train[i].flatten(),
        W,
        b
    )

    if prediction == y_train[i]:
        correct += 1

accuracy = correct / 1000

print("accuracy:", accuracy)