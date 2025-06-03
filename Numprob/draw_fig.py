import json
import matplotlib.pyplot as plt
import pdb

def statistic_modify_degree():
    with open(f"Numprob/intervene_results/modify_degree.json", 'r', encoding='utf-8') as file:
            data = json.load(file)
    
    count_dict = {
         '0.03': 0,
         '0.04': 0,
         '0.05': 0,
         '0.06': 0,
         '0.07': 0,
         '0.08': 0,
         '0.09': 0,
         '0.1': 0,
         '0.2': 0,
         '0.3': 0,
    }
    

    for data_index in data.keys():
        for layer_index in data[data_index].keys():
              for token_index in data[data_index][layer_index].keys():
                    if abs(data[data_index][layer_index][token_index]) < 0.03:
                        count_dict['0.03'] += 1 
                    elif abs(data[data_index][layer_index][token_index]) < 0.04:
                        count_dict['0.04'] += 1 
                    elif abs(data[data_index][layer_index][token_index]) < 0.05:
                        count_dict['0.05'] += 1 
                    elif abs(data[data_index][layer_index][token_index]) < 0.06:
                        count_dict['0.06'] += 1 
                    elif abs(data[data_index][layer_index][token_index]) < 0.07:
                        count_dict['0.07'] += 1 
                    elif abs(data[data_index][layer_index][token_index]) < 0.08:
                        count_dict['0.08'] += 1 
                    elif abs(data[data_index][layer_index][token_index]) < 0.09:
                        count_dict['0.09'] += 1 
                    elif abs(data[data_index][layer_index][token_index]) < 0.1:
                        count_dict['0.1'] += 1 
                    elif abs(data[data_index][layer_index][token_index]) < 0.2:
                        count_dict['0.2'] += 1 
                    elif abs(data[data_index][layer_index][token_index]) < 0.3:
                        count_dict['0.3'] += 1 

    print(count_dict)
    
    return count_dict

def plot_acc_json(file_path):

    with open(file_path, 'r') as f:
        data = json.load(f)
    
    x = [float(key) for key in data.keys()]  
    y = list(data.values()) 

    count_dict = statistic_modify_degree()
    counts = [count_dict[key] for key in count_dict.keys()]
    counts = [0] + counts
    max_count = counts[0]
    counts[1] = 1500

    fig, ax1 = plt.subplots(figsize=(10, 6))
    color1 = '#7E5CAD'
    ax1.plot(x, y, marker='o', linestyle='--', color=color1, label='Error ratio', markerfacecolor='white')
    ax1.set_xlabel("Intervene coefficient", fontsize=12)
    ax1.set_ylabel("Error ratio", fontsize=12, color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, linestyle='--', alpha=0.4)

    norm = plt.Normalize(min(y), max(y))
    alphas = 0.1 + 0.9 * norm(y) 

    color2 = '#2575ff'
    ax2 = ax1.twinx()
    bar_width = 0.01
    for xi, count, yi in zip(x, counts, y):
        if count == 1500:
            bar_width = 3*bar_width
            alphas[list(x).index(xi)] = 0.4
        ax2.bar(
            xi-bar_width/2, 
            count, 
            width=bar_width, 
            edgecolor=color2,  # Border color
            linewidth=1.5,  # Border width
            color=color2+'20', 
            # alpha = 0.3,
            label='Sample count' if xi == x[0] else ""  # 避免重复图例
        )
        ax2.text(
            xi - bar_width/2,  # x 位置（柱状图中心）
            count + max(counts)*0.02,  # y 位置（柱顶稍上方）
            f'{count}',                # 文本内容
            ha='center',               # 水平居中
            va='bottom',               # 垂直底部对齐
            fontsize=8,
            color='black'
        )
        bar_width = 0.01
        alphas = 0.1 + 0.9 * norm(y)
    ax2.set_ylabel("Count", fontsize=12, color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.set_ylim(0, max(counts)*1.1)
    

    target_key = 0.03
    if str(target_key) in data:
        target_value = data[str(target_key)]
        plt.axvline(x=target_key, color='r', linestyle='-', alpha=1, label=f'Threshold = {target_key}')
        # 在左侧坐标轴标注 value
        tag = plt.annotate(
            f'error ratio = {target_value:.3f}',
            xy=(target_key, target_value*1150/0.025),
            xytext=(-75, 10),    
            color=color1,
            textcoords='offset points',
            arrowprops=dict(arrowstyle='->'),  # 添加箭头
            bbox=dict(boxstyle='round', edgecolor='#7E5CAD', facecolor='white', alpha=1)  # 文本框样式
        )

    ax = plt.gca()
    # ax.spines['top'].set_visible(False)  # 上侧框线   

    ax1.set_zorder(2)  # ax1（折线图）在上层
    ax2.set_zorder(1)  # ax2（柱状图）在下层
    tag.set_zorder(2)
    ax1.set_frame_on(False)  # 避免ax1的边框遮挡ax2               
    
    # 获取两个图例的句柄和标签
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    # 创建合并的图例，并分两列显示
    legend = ax1.legend(
        lines1 + lines2, 
        labels1 + labels2,
        loc='lower right',  # 定位在右下角
        bbox_to_anchor=(0.95, 0.05),  # 微调位置
        ncol=2,  # 分两列显示
        frameon=True,  # 显示边框
        framealpha=0.8,  # 边框透明度
        borderpad=0.5  # 边框内边距
    )

    # 确保图例不会超出画布
    plt.tight_layout()

    plt.title("Intervene leads to calculation errors", fontsize=14)
    plt.xlabel("Intervene coefficient", fontsize=12)
    # plt.grid(True, linestyle='--', alpha=0.4)
    
    plt.savefig(f"Numprob/draw_curve/Acc_of_intervene.svg", bbox_inches='tight', dpi=300)
    plt.show()

# 调用函数（假设文件在当前目录下）
plot_acc_json("Numprob/intervene_results/acc.json")


