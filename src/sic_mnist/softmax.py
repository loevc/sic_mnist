import numpy as np


def softmax(x):
    # softmax(x - C) === softmax(x)
    x = x - np.max(x)
    exp_x = np.exp(x)

    return exp_x / np.sum(exp_x)


x = np.array([1.0, 2.0, 3.0])
# softmax 只关心相对大小，不关心绝对位置
# 差距被"放大"了
# x = np.array([3.0, 2.0, 1.0])

# 数值稳定性很重要，不然，本来很好计算的问题，却引发了计算失败
# x = np.array([1000, 1001, 1002])

y = softmax(x)

print(y)
print("sum =", np.sum(y))