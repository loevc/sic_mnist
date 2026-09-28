import numpy as np


def softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


# =========================
# 1. 准备数据
# =========================

x = np.random.rand(784)

label = 3


# =========================
# 2. 初始化参数
# =========================

W = np.random.randn(10, 784) * 0.01
b = np.zeros(10)


# =========================
# 3. Forward
# =========================

z = W @ x + b

p = softmax(z)

loss = -np.log(p[label])


print("========== Forward ==========")
print("x.shape    =", x.shape)
print("W.shape    =", W.shape)
print("b.shape    =", b.shape)
print("z.shape    =", z.shape)
print("p.shape    =", p.shape)

print()
print("label      =", label)
print("prediction =", np.argmax(p))
print("loss       =", loss)


# =========================
# 4. One-hot
# =========================

y = np.zeros(10)
y[label] = 1


# =========================
# 5. Backward
# =========================

dz = p - y

dW = np.outer(dz, x)

db = dz


print()
print("========== Backward ==========")
print("dz.shape =", dz.shape)
print("dW.shape =", dW.shape)
print("db.shape =", db.shape)


# =========================
# 6. Update
# =========================

learning_rate = 0.1

W = W - learning_rate * dW
b = b - learning_rate * db


print()
print("========== Update ==========")
print("W updated")
print("b updated")