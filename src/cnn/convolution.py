import numpy as np


def conv2d(x, kernel):

    h, w = x.shape
    kh, kw = kernel.shape

    out_h = h - kh + 1
    out_w = w - kw + 1

    output = np.zeros(
        (out_h, out_w)
    )

    for i in range(out_h):
        for j in range(out_w):

            region = x[
                i:i + kh,
                j:j + kw
            ]

            output[i, j] = np.sum(
                region * kernel
            )

    return output


x = np.array([
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25],
])

kernel = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1],
])

# 没有 padding、stride = 1
# output_size = input_size - kernel_size + 1
# H_out = (H + 2P - K) / S + 1
# W_out = (W + 2P - K) / S + 1
output = conv2d(x, kernel)

print(output)