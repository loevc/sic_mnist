import numpy as np


def softmax(x):
    # softmax(x - C) === softmax(x)
    x = x - np.max(x)
    exp_x = np.exp(x)

    return exp_x / np.sum(exp_x)


x = np.array([1.0, 2.0, 3.0])

y = softmax(x)

print(y)
print("sum =", np.sum(y))