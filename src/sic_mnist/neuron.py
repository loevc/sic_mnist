import numpy as np


# 模拟一张 MNIST 图片
x = np.random.rand(784)

# 10 个输出神经元
W = np.random.randn(10, 784)

# 10 个 bias
b = np.random.randn(10)


y = W @ x + b


print("x.shape =", x.shape)
print("W.shape =", W.shape)
print("b.shape =", b.shape)
print("y.shape =", y.shape)

print()
print("y =", y)
# print("x =", x)
# print("W =", W)
# print("b =", b)

