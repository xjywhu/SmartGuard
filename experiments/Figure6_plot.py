import json
# import matplotlib
# matplotlib.use('TkAgg')
import pickle

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# plt.rcParams['figure.figsize'] = 13, 6
length = 15


def plot_loss(data1, data2, data3):
    x = [i+1 for i in range(length)]

    y1 = data1
    y2 = data2
    y3 = data3

    # 创建画布和子图
    fig, ax = plt.subplots()

    # 绘制三条线
    ax.plot(x, y1, label='w/o Mask', linewidth=3)
    ax.plot(x, y2, label='Random Mask', linewidth=3)
    ax.plot(x, y3, label='Top-k Loss Mask', linewidth=3)
    # ax.plot(x, y4, label='Loss Guided Mask', linewidth=3)
    ax.tick_params(axis='both', which='major', labelsize=18)

    # 添加图例
    ax.legend(fontsize=15, frameon=False)
    xticks_values = [i+1 for i in range(length)]
    ax.set_xticks(xticks_values)
    # plt.title('Multiple Lines on a Single Plot')
    plt.xlabel('Training Step', fontsize=20)
    plt.ylabel('Mean of Reconstruction Loss', fontsize=20)
    # plt.ylabel('Variance')


    plt.tight_layout()
    plt.savefig("figure/compare_mean.pdf")
    plt.show()


def plot_var(data1, data2, data3):

    begin = 0
    end = length
    x = [i for i in range(1, end - begin + 1)]


    y1 = data1[begin:end]
    y2 = data2[begin:end]
    y3 = data3[begin:end]


    fig, ax = plt.subplots()

    ax.plot(x, y1, label='w/o Mask', linewidth=2.5)
    ax.plot(x, y2, label='Random Mask', linewidth=2.5)
    ax.plot(x, y3, label='Top-k Loss Mask', linewidth=2.5)

    ax.legend(loc="lower left", fontsize=15, frameon=False)

    xticks_values = [i + 1 for i in range(end - begin)]
    ax.set_xticks(xticks_values)
    ax.tick_params(axis='both', which='major', labelsize=18)

    # plt.title('Multiple Lines on a Single Plot')
    plt.xlabel('Training Step', fontsize=20)
    # plt.ylabel('Loss')
    plt.ylabel('Variance of Reconstruction Loss', fontsize=20)

    axins = inset_axes(ax, width='45%', height='45%', loc='upper right')

    begin2 = 8
    end2 = 15
    axins.plot(x[begin2:end2], y1[begin2:end2], linewidth=3)
    axins.plot(x[begin2:end2], y2[begin2:end2], linewidth=3)
    axins.plot(x[begin2:end2], y3[begin2:end2], linewidth=3)

    axins.set_xticks([9, 10, 11, 12, 13, 14, 15])
    axins.set_ylim(0, 1.5)

    # set box
    xlim0 = 9
    xlim1 = 15
    ylim0 = 0
    ylim1 = 1.2
    sx = [xlim0, xlim1, xlim1, xlim0, xlim0]
    sy = [ylim0, ylim0, ylim1, ylim1, ylim0]
    ax.plot(sx, sy, 'black')


    xy = (xlim0, ylim1)
    xy2 = (9-0.2, 0)
    con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
                          axesA=axins, axesB=ax, linestyle="--")
    axins.add_artist(con)

    xy = (xlim1, ylim0)
    xy2 = (15+0.3, 0)
    con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
                          axesA=axins, axesB=ax, linestyle="--")
    axins.add_artist(con)
    plt.tight_layout()
    plt.savefig("figure/compare_var.pdf")

    plt.show()


def get_data(file_name="results/MaskedAutoencoder_random.json"):
    vars = []
    losses = []

    with open(file_name, "r") as fs:
        res = json.loads(fs.read())
        # print(res)

        for item in res:
            losses.append(item['ALL Loss'])
            vars.append(item['Var'])
    return vars, losses


def plot_behavior_distribution(dataset):
    # plt.rcParams['figure.figsize'] = 9, 6
    train_file1 = f"data2/{dataset}_data/deleted_flattened_useful_{dataset}_trn_instance_10.pkl"
    with open(train_file1, "rb") as file:
        data = pickle.load(file)
    data = np.array(data)
    indices = np.arange(3, 40, 4)
    data = data[:, indices]
    behavior_dic = {}
    for item in data:
        for be in item:
            if be in behavior_dic:
                behavior_dic[be] += 1
            else:
                behavior_dic[be] = 1
    # print(behavior_dic)
    res = []
    for key in behavior_dic:
        res.append((behavior_dic[key], key))
    res = sorted(res)

    numbers = []
    for item in reversed(res):
        numbers.append(item[0])
    print(numbers)

    # 将行为出现的次数转换为概率分布
    # total_counts = sum(numbers)
    # probabilities = [count / total_counts for count in numbers]
    #
    # # 计算熵
    # entropy = -sum(p * np.log2(p) for p in probabilities)
    #
    # print("行为分布的熵：", entropy)

    # 计算总样本数量
    total_samples = sum(numbers)

    # 计算各类别样本比率
    class_ratios = [count / total_samples for count in numbers]

    # 计算基尼系数
    gini_coefficient = 1 - sum([p ** 2 for p in class_ratios])

    print("数据集的基尼系数：", gini_coefficient)

    fig, ax1 = plt.subplots()
    plt.tight_layout()

    # 绘制柱状图
    x = [i for i in range(len(numbers))]
    plt.bar(x, numbers, color='#a3c3f5')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('Behavior ID', fontsize=15)
    plt.ylabel('Number of Behaviors', fontsize=15)
    plt.tight_layout()
    # ax1.set_ylabel('Number of Behavior', fontsize=25)
    plt.savefig("./plot/{}_distribution.pdf".format(dataset))
    plt.show()





if __name__ == "__main__":
    vars1, losses1 = get_data(file_name="results/TransformerAutoencoder_random.json")
    vars2, losses2 = get_data(file_name="results/MaskedAutoencoder_random.json")
    vars3, losses3 = get_data(file_name="results/MaskedAutoencoder_top_k_loss.json")

    # vars, losses = get_data()
    # print(vars)
    # print(losses)

    plot_loss(losses1[0:length], losses2[0:length], losses3[0:length])
    plot_var(vars1[0:length], vars2[0:length], vars3[0:length])

    # plot_behavior_distribution("an")
    # plot_behavior_distribution("fr")
    # plot_behavior_distribution("sp")