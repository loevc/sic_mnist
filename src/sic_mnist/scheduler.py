import math

class WarmupCosineScheduler:

    def __init__(
        self,
        optimizer,
        warmup_epochs,
        total_epochs,
        max_lr,
        min_lr=0.0
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.max_lr = max_lr
        self.min_lr = min_lr

    def step(self, epoch):

        if epoch < self.warmup_epochs:

            lr = self.max_lr * (
                (epoch + 1) / self.warmup_epochs
            )

        else:

            progress = (
                epoch - self.warmup_epochs
            ) / (
                self.total_epochs - self.warmup_epochs
            )

            lr = self.min_lr + 0.5 * (
                self.max_lr - self.min_lr
            ) * (
                1 + math.cos(math.pi * progress)
            )

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

# import torch
# import torch.nn as nn
#
#
# model = nn.Sequential(
#     nn.Linear(784, 128),
#     nn.ReLU(),
#     nn.Linear(128, 10)
# )
#
# optimizer = torch.optim.SGD(
#     model.parameters(),
#     lr=0.0
# )
#
# scheduler = WarmupCosineScheduler(
#     optimizer=optimizer,
#     warmup_epochs=5,
#     total_epochs=20,
#     max_lr=0.1,
#     min_lr=0.001
# )
#
# for epoch in range(20):
#
#     scheduler.step(epoch)
#
#     print(
#         epoch + 1,
#         optimizer.param_groups[0]["lr"]
#     )