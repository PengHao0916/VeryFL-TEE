import re
import matplotlib.pyplot as plt
from collections import defaultdict
import os

# --- 配置 ---
# 请将下面的文件名替换为您上一步找到的、最新的那个日志文件名
LOG_FILE = 'log/2025_10_12_5.log'
# ---

def parse_log_file(file_path):
    """解析日志文件，提取每一轮的准确率和损失值。"""
    epoch_accuracies = defaultdict(list)
    epoch_losses = defaultdict(list)

    # 终极版正则表达式，精确匹配 "inner epoch X, Loss: Y, Acc: Z"
    log_pattern = re.compile(r"inner epoch (\d+), Loss: ([\d.]+), Acc: ([\d.]+)")

    if not os.path.exists(file_path):
        print(f"错误：文件 '{file_path}' 未找到。")
        print("请确保上面的 LOG_FILE 变量已设置为正确的、最新的日志文件名。")
        return None, None

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = log_pattern.search(line)
            if match:
                epoch = int(match.group(1))
                loss = float(match.group(2))
                accuracy = float(match.group(3))

                epoch_losses[epoch].append(loss)
                epoch_accuracies[epoch].append(accuracy)

    if not epoch_accuracies:
        print("错误：未能在日志文件中找到任何有效的准确率数据。")
        print("请仔细检查日志文件内容和 LOG_FILE 变量是否正确。")
        return None, None

    avg_accuracies = {epoch: sum(accs) / len(accs) for epoch, accs in epoch_accuracies.items()}
    avg_losses = {epoch: sum(losses) / len(losses) for epoch, losses in epoch_losses.items()}

    return avg_accuracies, avg_losses


def plot_metrics(accuracies, losses, file_name):
    """绘制并保存准确率和损失值曲线图。"""
    if not accuracies or not losses:
        print("没有可供绘制的数据。")
        return

    epochs = sorted(accuracies.keys())
    acc_values = [accuracies[epoch] for epoch in epochs]
    loss_values = [losses[epoch] for epoch in epochs]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle(f'Federated Learning Performance\nLog File: {file_name}', fontsize=16)

    ax1.plot(epochs, acc_values, 'b-o', markersize=4, label='Average Accuracy')
    ax1.set_title('Model Accuracy vs. Communication Rounds')
    ax1.set_xlabel('Communication Round (Epoch)')
    ax1.set_ylabel('Accuracy (%)')
    ax1.grid(True)
    ax1.legend()

    ax2.plot(epochs, loss_values, 'r-o', markersize=4, label='Average Loss')
    ax2.set_title('Model Loss vs. Communication Rounds')
    ax2.set_xlabel('Communication Round (Epoch)')
    ax2.set_ylabel('Loss')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_filename = 'experiment_results.png'
    plt.savefig(output_filename)

    print(f"\n图表已成功保存为 '{output_filename}'")
    if acc_values:
        print(f"最终平均准确率: {acc_values[-1]:.2f}%")


if __name__ == '__main__':
    try:
        import matplotlib
    except ImportError:
        print("错误：Matplotlib 未安装。请运行 'pip install matplotlib' 进行安装。")
    else:
        accuracies, losses = parse_log_file(LOG_FILE)
        plot_metrics(accuracies, losses, LOG_FILE)