import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体以避免字体缺失警告
plt.rcParams['font.sans-serif']=['Arial Unicode MS']

# 所有模型名称（原始模型 + 5 个新模型）
models = ['原始模型', 'TF-IDF+SVC', 'TF-IDF+LR', 'Hashing+SGD', 'TF-IDF+NB', 'W2V+LR']

# 场景 1：小数据集 dataset.txt mode1（原始模型 + 5 个新模型）
times_1 = [309, 1.46, 2.64, 0.92, 2.62, 6.05]     # 时间（秒）
accs_1  = [0.913, 0.9928, 0.9891, 0.9923, 0.9901, 0.9780]
f1s_1   = [0.895, 0.9916, 0.9873, 0.9909, 0.9885, 0.9746]

# 场景 2：大数据集 big_dataset.txt mode1（原始模型 + 5 个新模型）
times_2 = [1380, 29.07, 41.02, 13.90, 49.52, 98.85]
accs_2  = [0.964, 0.9941, 0.9896, 0.9810, 0.9694, 0.9471]
f1s_2   = [0.891, 0.9836, 0.9704, 0.9429, 0.9024, 0.8404]

# 场景 3：mode2: 用 big_dataset.txt 训练, dataset.txt 测试（原始模型 + 5 个新模型）
times_3 = [1834, 39.24, 48.11, 15.52, 68.47, 169.56]
accs_3  = [0.435, 0.6614, 0.5825, 0.4134, 0.3826, 0.4280]
f1s_3   = [0.360, 0.6566, 0.5823, 0.3940, 0.3441, 0.4051]

# 颜色列表：确保原始模型与后续模型在图中拥有不同颜色
colors_time = ['#d62728', '#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b']
colors_acc  = ['#bcbd22', '#17becf', '#e377c2', '#7f7f7f', '#aec7e8', '#ffbb78']
colors_f1   = ['#2ca02c', '#d62728', '#1f77b4', '#ff7f0e', '#17becf', '#9467bd']

def plot_combined(title, times, accs, f1s):
    x = np.arange(len(models))
    width = 0.2

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 时间条, 使用对应颜色
    for i in range(len(models)):
        ax1.bar(x[i] - width, times[i], width * 0.9, color=colors_time[i], label='时间 (s)' if i == 0 else "")

    # Accuracy 条，共享 x 轴，用第二 y 轴
    ax2 = ax1.twinx()
    for i in range(len(models)):
        ax2.bar(x[i], accs[i], width * 0.9, color=colors_acc[i], label='Accuracy' if i == 0 else "")

    # Macro-F1 条，用第三 y 轴并向右偏移
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("outward", 60))
    for i in range(len(models)):
        ax3.bar(x[i] + width, f1s[i], width * 0.9, color=colors_f1[i], label='Macro-F1' if i == 0 else "")

    # X 轴设置
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.set_title(title, fontsize=14)

    # Y 轴标签
    ax1.set_ylabel('时间 (秒)', color='black')
    ax2.set_ylabel('Accuracy', color='black')
    ax3.set_ylabel('Macro-F1', color='black')

    # 自定义图例：手动创建三种指标的图例
    handles = [
        plt.Line2D([0], [0], color=colors_time[0], lw=8),
        plt.Line2D([0], [0], color=colors_acc[0], lw=8),
        plt.Line2D([0], [0], color=colors_f1[0], lw=8)
    ]
    labels = ['时间 (s)', 'Accuracy', 'Macro-F1']
    ax1.legend(handles, labels, loc='upper left')

    fig.tight_layout()

# 绘制三种场景对比图
plot_combined('场景1：dataset.txt (mode1) 原始 vs 5个模型', times_1, accs_1, f1s_1)
plot_combined('场景2：big_dataset.txt (mode1) 原始 vs 5个模型', times_2, accs_2, f1s_2)
plot_combined('场景3：mode2 (big→dataset) 原始 vs 5个模型', times_3, accs_3, f1s_3)

plt.show()