import torch
import torch.nn as nn

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

model = model.to(device)

print(model)

criterion = nn.CrossEntropyLoss()

x = torch.randn(32, 784)

labels = torch.randint(
    0,
    10,
    (32,)
)

x = x.to(device)
labels = labels.to(device)


output = model(x)

print("output:", output.shape)

loss = criterion(
    output,
    labels
)

print("loss:", loss.item())


optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1
)


optimizer.zero_grad()

# output = model(x)
#
# loss = criterion(output, labels)

loss.backward()

optimizer.step()