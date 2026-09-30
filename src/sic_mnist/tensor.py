import torch

x = torch.tensor(
    [1, 2, 3],
    dtype=torch.float32
)

print(x)
print(type(x))

# grad
x = torch.tensor(
    2.0,
    requires_grad=True
)

y = x ** 2

y.backward()

print(x.grad)