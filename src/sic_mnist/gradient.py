import numpy as np


def f(x):
    return x ** 2


def gradient(x):
    return 2 * x


x = 5.0
learning_rate = 0.1

for step in range(20):
    y = f(x)
    grad = gradient(x)

    print(
        f"step={step:2d}, "
        f"x={x:.6f}, "
        f"y={y:.6f}, "
        f"gradient={grad:.6f}"
    )

    x = x - learning_rate * grad