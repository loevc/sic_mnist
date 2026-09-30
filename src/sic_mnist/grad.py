import torch

x = torch.tensor(
    2.0,
    # 形成计算图
    requires_grad=True
)

a = x * 3
b = a + 1
y = b ** 2

y.backward()

print("x:", x)
print("a:", a)
print("b:", b)
print("y:", y)

print("dy/dx:", x.grad)