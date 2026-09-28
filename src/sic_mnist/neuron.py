import numpy as np


# 3 个输入
x = np.array([2.0, 3.0, 4.0])

# 3 个权重
w = np.array([0.1, 0.2, 0.3])

# bias
b = 0.5


# 神经元计算
y = np.dot(w, x) + b

print("x =", x)
print("w =", w)
print("b =", b)
print("y =", y)