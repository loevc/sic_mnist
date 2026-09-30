import time

import torch
import mnist_reader
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import argparse

from logger import log
log.set_level(log.INFO)


# ============================================================
# 0. Args
# ============================================================

parser = argparse.ArgumentParser()
DEFAULT_LR = 0.369
parser.add_argument("--lr", type=float, required=False, default=DEFAULT_LR, help="learning rate for SGD")
args = parser.parse_args()
LR = args.lr

# ============================================================
# 1. Device
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

log.info("device:", device)

if torch.cuda.is_available():
    log.debug("GPU:", torch.cuda.get_device_name(0))

# ============================================================
# 2. NumPy -> Tensor
# ============================================================


X_train, y_train, X_test, y_test = mnist_reader.read_data_sets_reshape()

X_train_tensor = torch.tensor(
    X_train,
    # W b 激活值 logits gradient 都是float
    dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train,
    # 这种分类任务中要求 label 是类别索引
    dtype=torch.long
)

X_test_tensor = torch.tensor(
    X_test,
    dtype=torch.float32
)

y_test_tensor = torch.tensor(
    y_test,
    dtype=torch.long
)

# ============================================================
# 3. Dataset
# ============================================================

train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor
)

test_dataset = TensorDataset(
    X_test_tensor,
    y_test_tensor
)

# ============================================================
# 4. DataLoader
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    # 每个 epoch 都把训练数据重新打乱
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

# ============================================================
# 5. Model
# ============================================================

model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
).to(device)

for name, param in model.named_parameters():
    log.debug(name)
    log.debug("shape:", param.shape)
    log.debug("requires_grad:", param.requires_grad)

# ============================================================
# 6. Loss
# ============================================================

criterion = nn.CrossEntropyLoss()

# ============================================================
# 7. Optimizer
# ============================================================

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=LR,
    # momentum=0.9
)

# optimizer = torch.optim.Adam(
#     model.parameters(),
#     lr=0.001
# )

# ============================================================
# 8. Training
# ============================================================

epochs = 5

start_time = time.time()

for epoch in range(epochs):

    # --------------------------------------------
    # Training mode
    # --------------------------------------------

    # Dropout BatchNorm, 未来会遇到
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        # 参与同一次计算的 Tensor 和模型参数必须在同一个 device 上
        # tensor : data,  模型参数： model
        images = images.to(device)
        labels = labels.to(device)

        old_weight = model[0].weight.clone()

        # 清空上一批次的 gradient, pytorch模型梯度累加
        optimizer.zero_grad()

        # forward
        logits = model(images)

        # loss
        loss = criterion(
            logits,
            labels
        )

        # view model
        for name, param in model.named_parameters():
            log.debug(name, param.grad)

        # backward
        loss.backward()

        # view model
        for name, param in model.named_parameters():
            log.debug(
                name,
                param.grad.shape,
                "gradient mean =",
                param.grad.mean().item()
            )

        # update parameters
        optimizer.step()

        new_weight = model[0].weight

        log.debug(
            "weight changed:",
            not torch.equal(
                old_weight,
                new_weight
            )
        )

        # ----------------------------------------
        # statistics
        # ----------------------------------------

        batch_size = images.size(0)

        total_loss += (
            loss.item() * batch_size
        )

        predictions = logits.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += batch_size

    train_loss = total_loss / total
    train_acc = correct / total


    # --------------------------------------------
    # Evaluation
    # --------------------------------------------

    model.eval()

    test_correct = 0
    test_total = 0

    # 测试阶段不需要计算梯度， 没必要构建完整的计算图，这样可以减少显存，减少计算开销，加快推理
    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)

            predictions = logits.argmax(
                dim=1
            )

            test_correct += (
                predictions == labels
            ).sum().item()

            test_total += labels.size(0)

    test_acc = test_correct / test_total


    log.info(
        f"epoch={epoch + 1}, "
        f"loss={train_loss:.4f}, "
        f"train_acc={train_acc:.4f}, "
        f"test_acc={test_acc:.4f}"
    )


elapsed = time.time() - start_time

log.info(f"cost: {elapsed:.4f} s")

# 2.30 ≈ ln(10)