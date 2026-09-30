import subprocess
import csv
import re
import numpy as np
import matplotlib.pyplot as plt

# ===================== 配置区 =====================
LR_MIN = 0.03
LR_MAX = 1.0
SAMPLE_COUNT = 10   # 0.001~2.0一共采多少个学习率点
# lr_list = np.linspace(LR_MIN, LR_MAX, SAMPLE_COUNT).tolist()
lr_list = np.logspace(np.log10(LR_MIN), np.log10(LR_MAX), SAMPLE_COUNT).tolist()


output_csv = "lr_experiment_result.csv"
output_fig = "lr_compare.png"
# ==================================================


all_records = []

# 正则提取 epoch loss train_acc test_acc
pattern = re.compile(r"epoch=(?P<epoch>\d+), loss=(?P<loss>[\d.]+), train_acc=(?P<train_acc>[\d.]+), test_acc=(?P<test_acc>[\d.]+)")

for lr in lr_list:
    print(f"\n===== uv run mnist.py , lr={lr} =====")
    proc = subprocess.Popen(
        [
            "uv", "run", "python",
            "mnist.py",
            "--lr", str(lr)
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        bufsize=1
    )

    while True:
        line = proc.stdout.readline()
        if not line and proc.poll() is not None:
            break
        if not line:
            continue
        print(line, end="")
        match = pattern.search(line)
        if match:
            d = match.groupdict()
            record = {
                "lr": float(lr),
                "epoch": int(d["epoch"]),
                "train_loss": float(d["loss"]),
                "train_acc": float(d["train_acc"]),
                "test_acc": float(d["test_acc"])
            }
            all_records.append(record)

# 写入csv
with open(output_csv, "w", encoding="utf-8", newline="") as f:
    fieldnames = ["lr", "epoch", "train_loss", "train_acc", "test_acc"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_records)

print(f"\n✅all finished，res output--> {output_csv}，collect {len(all_records)} records")


# ===================== 绘图部分 =====================
if not all_records:
    print("⚠no collect data，skip plt")
else:
    # 按 lr 分组
    from collections import defaultdict
    data_by_lr = defaultdict(lambda: {"epoch":[], "test_acc":[]})
    for r in all_records:
        data_by_lr[r["lr"]]["epoch"].append(r["epoch"])
        data_by_lr[r["lr"]]["test_acc"].append(r["test_acc"])

    plt.figure(figsize=(10,6), dpi=120)
    for lr, v in data_by_lr.items():
        plt.plot(v["epoch"], v["test_acc"], marker="o", label=f"lr={lr:.4f}")

    plt.xlabel("epoch")
    plt.ylabel("test_acc")
    plt.title(f"LR compare ({LR_MIN} ~ {LR_MAX}), test accuracy over epoch")
    plt.legend(loc="best", fontsize=8)
    plt.grid(alpha=0.3)
    plt.xticks(np.arange(1, max(v["epoch"]) +1, 1))
    plt.tight_layout()
    plt.savefig(output_fig)
    print(f"📈curve save --> {output_fig}")
    plt.show()
