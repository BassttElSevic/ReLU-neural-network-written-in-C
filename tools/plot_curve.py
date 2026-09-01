# 训练可视化脚本
# 读取 data/training_curve.csv,绘制损失曲线和准确率曲线(中英双语标签)。
# 依赖: python3 + matplotlib + numpy ,系统需装有中文字体(如文泉驿微米黑)。
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 训练产物目录。所有读写都约束在这个固定目录下,不使用外部传入的任意路径。
DATA_DIR = "data"


def data_path(filename):
    """拼接 data/ 目录下的路径。只接受文件名,拒绝包含路径分隔符的输入。"""
    if os.sep in filename or (os.altsep and os.altsep in filename):
        raise ValueError("data_path 只接受文件名,不接受路径")
    return os.path.join(DATA_DIR, filename)


def pick_cjk_font():
    """从系统已安装字体里挑一个可用的中文 sans 字体,找不到则回退默认。"""
    candidates = [
        "WenQuanYi Micro Hei",      # 文泉驿微米黑
        "Noto Sans CJK SC",         # 思源黑体简体
        "Source Han Sans SC",       # 思源黑体
        "Microsoft YaHei",
        "PingFang SC",
        "Heiti SC",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    # 兜底: 用 findfont 尽力解析一个存在的中文字体
    try:
        return fm.findfont("WenQuanYi Micro Hei", fallback_to_default=True)
    except Exception:
        return "DejaVu Sans"


def load_csv():
    """读取固定的 data/training_curve.csv,返回 (epochs, loss, tacc, vacc)。"""
    path = data_path("training_curve.csv")
    epochs, loss, tacc, vacc = [], [], [], []
    try:
        with open(path, "r") as f:
            f.readline()  # 跳过表头
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                epochs.append(int(parts[0]))
                loss.append(float(parts[1]))
                tacc.append(float(parts[2]))
                vacc.append(float(parts[3]))
    except (IOError, ValueError, IndexError) as e:
        print(f"读取 {path} 失败: {e}")
        sys.exit(1)
    return epochs, loss, tacc, vacc


def main():
    # 让图表文字不出现方框乱码
    plt.rcParams["font.sans-serif"] = [pick_cjk_font(), "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    csv_path = data_path("training_curve.csv")
    if not os.path.exists(csv_path):
        print(f"未找到 {csv_path},请先运行训练程序生成曲线数据。")
        sys.exit(1)

    epochs, loss, tacc, vacc = load_csv()
    if not epochs:
        print(f"{csv_path} 中没有可用的数据行。")
        sys.exit(1)

    plt.figure(figsize=(11.5, 4.5))

    # 左图: 训练损失(期望单调下降)
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, color="#1f77b4", linewidth=2, label="训练损失 training loss")
    plt.xlabel("迭代轮数 epoch")
    plt.ylabel("交叉熵损失 cross-entropy loss")
    plt.title("训练损失曲线 Training Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 右图: 训练/验证准确率(期望趋近 1.0)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, tacc, color="#2ca02c", linewidth=2, label="训练准确率 train accuracy")
    plt.plot(epochs, vacc, color="#d62728", linestyle="--", linewidth=2, label="验证准确率 val accuracy")
    plt.xlabel("迭代轮数 epoch")
    plt.ylabel("准确率 accuracy")
    plt.title("准确率曲线 Accuracy")
    plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    save_csv_path = data_path("training_curve.png")
    try:
        plt.savefig(save_csv_path, dpi=150)
    except (IOError, OSError) as e:
        print(f"保存图片失败: {e}")
        sys.exit(1)
    print(f"图表已保存到 {save_csv_path}")


if __name__ == "__main__":
    main()
