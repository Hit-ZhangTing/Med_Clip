import os
import re
import matplotlib.pyplot as plt

def plot_loss_from_log(log_path="checkpoints/train.log", save_path="loss_curve.png"):
    if not os.path.exists(log_path):
        print(f"找不到日志文件: {log_path}")
        return

    train_epochs, train_losses = [], []
    val_epochs, val_losses = [], []

    # 解析日志正则
    # 匹配: [Epoch 1/30] Train Loss: 7.2803
    train_pattern = re.compile(r"\[Epoch\s+(\d+)/\d+\]\s+Train Loss:\s+([\d\.]+)")
    # 匹配: [Epoch 1] Val Loss: 14.1203
    val_pattern = re.compile(r"\[Epoch\s+(\d+)\]\s+Val Loss:\s+([\d\.]+)")

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            t_match = train_pattern.search(line)
            if t_match:
                train_epochs.append(int(t_match.group(1)))
                train_losses.append(float(t_match.group(2)))
                
            v_match = val_pattern.search(line)
            if v_match:
                val_epochs.append(int(v_match.group(1)))
                val_losses.append(float(v_match.group(2)))

    # 如果没数据就不画
    if not train_epochs:
        print("日志里还没有产生完整的 Epoch Loss 数据，再等等...")
        return

    # 开始画图
    plt.figure(figsize=(10, 6))
    plt.plot(train_epochs, train_losses, label="Train Loss", marker="o", color="#1f77b4", linewidth=2)
    
    if val_epochs:
        plt.plot(val_epochs, val_losses, label="Validation Loss", marker="s", color="#ff7f0e", linewidth=2)

    plt.title("Ultrasound-CLIP V2 Training Curve", fontsize=14, fontweight="bold")
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Total Loss", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    # 保存图片
    plt.savefig(save_path, dpi=300)
    print(f"➡️ 监控曲线已更新！请查看刚才生成的图片: {save_path}")


if __name__ == "__main__":
    plot_loss_from_log()
