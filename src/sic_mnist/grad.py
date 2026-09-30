import torch

x = torch.tensor(
    2.0,
    # 形成计算图
    requires_grad=True
)

y = x ** 3

print("x =", x)
print("y =", y)

y.backward()

print(x.grad)