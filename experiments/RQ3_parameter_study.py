import json
import numpy as np

mask_ratio_dic = {0.2: 0, 0.4: 1, 0.6: 2, 0.8: 3}
# mask_step_dic = {2: 0, 3: 1, 4: 2, 5: 3}
# mask_step_dic = {0: 0, 5: 1, 10: 2, 15: 3}

mask_step_dic = {3: 0, 4: 1, 5: 2, 6: 3}
layer_dic = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}
# embedding_dic = {128: 0, 256: 1, 512: 2, 1024: 3}
embedding_dic = {8: 0, 16: 1, 32: 2, 64: 3, 128: 4, 256: 5, 512: 6, 1024: 7}

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


# plt.rcParams['figure.figsize'] = 13, 6


def plot(x, y1, y2, y3, y4):
    end = 7
    x_values = x[0:end]

    plt.plot(x_values, y1[0:end], marker='o', linestyle='--', markersize=10, linewidth=3, label='SD')
    plt.plot(x_values, y2[0:end], marker='^', linestyle='--', markersize=10, linewidth=3, label='MD')
    plt.plot(x_values, y3[0:end], marker='s', linestyle='--', markersize=10, linewidth=3, label='DM')
    plt.plot(x_values, y4[0:end], marker='*', linestyle='--', markersize=10, linewidth=3, label='DD')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim([0.95, 1])

    # 添加标题和标签
    # plt.title('Example Line Plot', fontsize=16)
    plt.xlabel('Dimension of Embeddings', fontsize=20)
    plt.ylabel('F1-Score', fontsize=20)
    plt.legend(fontsize=20, frameon=False)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./figure/em.pdf")
    plt.show()


def plot_layer():
    with open("results/layer_para_an.json", "r") as file_para:
        res = json.loads(file_para.read())

    data_dic = {'SD': np.zeros(6), 'MD': np.zeros(6), 'DM': np.zeros(6),
                'DD': np.zeros(6)}
    for item in res:
        attack_type = item['type']
        data_dic[attack_type][layer_dic[item['layer']]] = item['f1_score']

    y1 = data_dic['SD']
    y2 = data_dic['MD']
    y3 = data_dic['DM']
    y4 = data_dic['DD']
    x = list(layer_dic)
    # 创建示例数据
    end = 4
    x_values = [1, 2, 3, 4]

    # 绘制折线图
    plt.plot(x_values, y1[0:end], marker='o', linestyle='--', markersize=10, linewidth=3, label='SD')
    plt.plot(x_values, y2[0:end], marker='^', linestyle='--', markersize=10, linewidth=3, label='MD')
    plt.plot(x_values, y3[0:end], marker='s', linestyle='--', markersize=10, linewidth=3, label='DM')
    plt.plot(x_values, y4[0:end], marker='*', linestyle='--', markersize=10, linewidth=3, label='DD')
    plt.xticks([1, 2, 3, 4], fontsize=20)
    plt.yticks(fontsize=20)
    # plt.ylim([0.984,1])

    # 添加标题和标签
    # plt.title('Example Line Plot', fontsize=16)
    plt.xlabel('layers of Encoder/Decoder', fontsize=20)
    plt.ylabel('F1-Score', fontsize=20)
    plt.legend(fontsize=20, frameon=False)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./figure/layer_para.pdf")
    plt.show()


# def plot_batch(x, y1, y2, y3, y4):
def plot_batch():

    x_values = [128, 256, 512, 1024, 2048]

    y1 = [0.9921568627450981, 0.9941291585127201, 0.9941176470588236, 0.9967916666666667, 0.9928617780661908]
    y2 = [0.981243830207305, 0.9822512315270936, 0.982283464566929, 0.9832507374631269, 0.9794319294809011]
    y3 = [0.9892262487757102, 0.990215264187867, 0.9912195121951219, 0.9941707317073171, 0.9883495145631067]
    y4 = [0.990215264187867, 0.9921875, 0.9922027290448342, 0.9951520467836257, 0.9893307468477206]

    plt.plot(x_values, y1, marker='o', linestyle='--', markersize=10, linewidth=3, label='SD')
    plt.plot(x_values, y2, marker='^', linestyle='--', markersize=10, linewidth=3, label='MD')
    plt.plot(x_values, y3, marker='s', linestyle='--', markersize=10, linewidth=3, label='DM')
    plt.plot(x_values, y4, marker='*', linestyle='--', markersize=10, linewidth=3, label='DD')
    plt.xticks([128, 512, 1024, 2048], fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim([0.97, 1])


    plt.xlabel('Batch Size', fontsize=20)
    plt.ylabel('F1-Score', fontsize=20)
    plt.legend(fontsize=20, frameon=False)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./figure/batch_para.pdf")
    plt.show()


def plot_lr():


    x_values = [1, 0.1, 0.01, 0.001, 0.0001]

    y1 = [0.3724867724867725, 0.8191489361702128, 0.931129476584022, 0.99609375, 0.9954456733897202]
    y2 = [0.03816793893129771, 0.36106750392464676, 0.33811802232854865, 0.9852507374631269, 0.9680638722554891]
    y3 = [0.03441682600382409, 0.3888888888888889, 0.3486529318541997, 0.9931707317073171, 0.9812807881773399]
    y4 = [0.05671077504725898, 0.6649616368286445, 0.6589446589446588, 0.9941520467836257, 0.9931840311587147]

    # 绘制折线图
    plt.plot(x_values, y1, marker='o', linestyle='--', markersize=10, linewidth=3, label='SD')
    plt.plot(x_values, y2, marker='^', linestyle='--', markersize=10, linewidth=3, label='MD')
    plt.plot(x_values, y3, marker='s', linestyle='--', markersize=10, linewidth=3, label='DM')
    plt.plot(x_values, y4, marker='*', linestyle='--', markersize=10, linewidth=3, label='DD')
    plt.xscale('log')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # plt.ylim([0.97, 1])

    # 添加标题和标签
    # plt.title('Example Line Plot', fontsize=16)
    plt.xlabel('Learning Rate', fontsize=20)
    plt.ylabel('F1-Score', fontsize=20)
    plt.legend(fontsize=20, frameon=False)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./figure/lr_para.pdf")
    plt.show()


def plot_heat_map(data, type):
    sns.heatmap(data, annot=True, fmt=".4f", cmap="YlGnBu", cbar=True, annot_kws={'size': 14})
    plt.xticks(np.arange(len(data)) + 0.5, labels=["3", "4", "5", "6"], fontsize=15)
    plt.yticks(np.arange(len(data)) + 0.5, labels=["0.2", "0.4", "0.6", "0.8"], fontsize=15)

    plt.xlabel("Step w/o Mask", fontsize=15)
    plt.ylabel("Mask Ratio", fontsize=15)
    plt.tight_layout()
    plt.savefig("./figure/{}.pdf".format(type))
    plt.show()



def plot_mask_var():
    y4 = [0.693172767, 0.482193437, 0.381485386, 0.249479811]
    y2 = [0.639924555, 0.428798293, 0.261389587, 0.216221541]
    y3 = [0.750938784, 0.556363322, 0.350005772, 0.247174107]
    y1 = [0.57213829, 0.398875441, 0.22917094, 0.14579382]

    labels = ['10', '11', '12', '13']

    # 设置柱状图宽度
    bar_width = 0.2

    # 设置x轴刻度位置
    r1 = np.arange(len(y1))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    r4 = [x + bar_width for x in r3]

    # 创建柱状图
    # plt.bar(r1, y1, width=bar_width,color='#81c9bc',edgecolor='black', label='Loss-guided Dynamic Mask')
    # plt.bar(r2, y2, width=bar_width,color='#e4cc8c',edgecolor='black', label='Top-k Loss Mask')
    # plt.bar(r3, y3, width=bar_width, color='#aee2ff',edgecolor='black', label='Random Mask')
    # plt.bar(r4, y4, width=bar_width,color='#e5e0ff', edgecolor='black', label='w/o Mask')

    plt.bar(r1, y1, width=bar_width, color='#f39fa1', edgecolor='black', label='LDMS')
    plt.bar(r2, y2, width=bar_width, color='#ffe699', edgecolor='black', label='Top-k Loss Mask')
    plt.bar(r3, y3, width=bar_width, color='#c5e0b4', edgecolor='black', label='Random Mask')
    plt.bar(r4, y4, width=bar_width, color='#a3c2f4', edgecolor='black', label='w/o Mask')

    # 添加标签
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Reconstruction Loss Variance', fontsize=20)
    plt.xticks([r + bar_width for r in range(len(y1))], labels, fontsize=20)
    plt.yticks(fontsize=20)
    # plt.ylim([0.982, 1])
    # 添加y轴标签
    plt.grid(axis='y', linestyle='-')
    # 添加图例
    plt.legend(ncol=1, fontsize=15, frameon=False, columnspacing=0.2)
    plt.tight_layout()
    # plt.savefig("./plot/mask_var.pdf")
    # 显示图表
    plt.show()


def plot_mask():
    # 数据
    y4 = [0.9935, 0.9833, 0.9902, 0.9901]
    y3 = [0.9941, 0.9833, 0.9902, 0.9912]
    y2 = [0.9947, 0.9833, 0.9912, 0.9922]
    y1 = [0.9967, 0.9834, 0.9941, 0.9951]

    labels = ['SD', 'MD', 'DM', 'DD']

    # 设置柱状图宽度
    bar_width = 0.2

    # 设置x轴刻度位置
    r1 = np.arange(len(y1))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    r4 = [x + bar_width for x in r3]

    # 创建柱状图
    # plt.bar(r1, y1, width=bar_width,color='#81c9bc',edgecolor='black', label='Loss-guided Dynamic Mask')
    # plt.bar(r2, y2, width=bar_width,color='#e4cc8c',edgecolor='black', label='Top-k Loss Mask')
    # plt.bar(r3, y3, width=bar_width, color='#aee2ff',edgecolor='black', label='Random Mask')
    # plt.bar(r4, y4, width=bar_width,color='#e5e0ff', edgecolor='black', label='w/o Mask')

    plt.bar(r1, y1, width=bar_width, color='#f39fa1', edgecolor='black', label='LDMS')
    plt.bar(r2, y2, width=bar_width, color='#ffe699', edgecolor='black', label='Top-k Loss Mask')
    plt.bar(r3, y3, width=bar_width, color='#c5e0b4', edgecolor='black', label='Random Mask')
    plt.bar(r4, y4, width=bar_width, color='#a3c2f4', edgecolor='black', label='w/o Mask')

    # 添加标签
    plt.xlabel('Anomaly Type', fontsize=20)
    plt.ylabel('F1-Score', fontsize=20)
    plt.xticks([r + bar_width for r in range(len(y1))], labels, fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim([0.982, 1])
    # 添加y轴标签
    plt.grid(axis='y', linestyle='-')
    # 添加图例
    plt.legend(ncol=2, fontsize=15, frameon=False, columnspacing=0.2)
    plt.tight_layout()
    # plt.savefig("./plot/mask_performance.pdf")
    # 显示图表
    plt.show()




def plot_mask_loss_compare():
    with open("results/loss_dic_sp.json") as data_file:
        data_dic = json.loads(data_file.read())
    plt.rcParams['figure.figsize'] = 13, 6
    # plt.rcParams['figure.figsize'] = 10, 6
    # 柱状图数据

    y1 = data_dic['num']

    y2 = data_dic['no_mask']
    # y3 = data_dic['random']
    y3 = data_dic['autoencoder']
    y4 = data_dic['gruautoencoder']
    y5 = data_dic['loss_guided']

    y2_new = []
    y1_new = []
    for (i, item) in enumerate(y2):
        # if item > 0.02:
        #     continue
        y1_new.append(y1[i])
        y2_new.append(y2[i])

    x = [i for i in range(len(y1_new))]
    y1 = y1_new
    y2 = y2_new

    fig, ax1 = plt.subplots()
    plt.tight_layout()


    ax1.bar(x, y1, color='#a3c3f5')
    ax1.set_ylabel('Number of Behavior', fontsize=25)

    ax2 = ax1.twinx()

    ax2.plot(x, y3, marker='o', markersize=8, label='Autoencoder')
    ax2.plot(x, y4, marker='o', markersize=8, label='ARGUS')
    ax2.plot(x, y2, color='#c00001', marker='o', markersize=8, label='TransAE')
    ax2.plot(x, y5, marker='o', markersize=8, label='SmartGuard')
    ax2.set_ylabel('Reconstruction Loss', fontsize=25)

    legend = ax2.legend(loc='best', prop={'size': 25})
    legend.get_frame().set_linewidth(0.0)
    # legend.get_texts().set_fontsize(20)
    # ax2.axhline(y=0.5, color='gray', linestyle='--', linewidth=4)

    # ax2.set_ylim(0, 0.02)

    # 添加标题和标签
    # plt.title('双坐标系示例')
    ax1.set_xlabel('Behavior ID', fontsize=25)

    ax1.tick_params(axis='both', which='major', labelsize=25)
    # ax2.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax2.tick_params(axis='both', which='major', labelsize=25)

    plt.tight_layout()
    plt.savefig("figure/unbalance_sp2.pdf")
    plt.show()

def plot_mu_fnr():
    mu1 = [0.01764, 0.01764, 0.01764, 0.01764]
    mu2 = [0.01964, 0.01964, 0.01964, 0.01964]
    mu3 = [0.02164, 0.02164, 0.02164, 0.02164]
    mu4 = [0.02164, 0.02164, 0.02164, 0.02164]
    labels = ['SD', 'MD', 'DM', 'DD']


    bar_width = 0.2


    r1 = np.arange(len(mu1))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    r4 = [x + bar_width for x in r3]

    # plt.bar(r1, y1, width=bar_width,color='#81c9bc',edgecolor='black', label='Loss-guided Dynamic Mask')
    # plt.bar(r2, y2, width=bar_width,color='#e4cc8c',edgecolor='black', label='Top-k Loss Mask')
    # plt.bar(r3, y3, width=bar_width, color='#aee2ff',edgecolor='black', label='Random Mask')
    # plt.bar(r4, y4, width=bar_width,color='#e5e0ff', edgecolor='black', label='w/o Mask')

    plt.bar(r1, mu1, width=bar_width, color='#f39fa1', edgecolor='black', label='$\mu=$100')
    plt.bar(r2, mu2, width=bar_width, color='#ffe699', edgecolor='black', label='$\mu=$10')
    plt.bar(r3, mu3, width=bar_width, color='#c5e0b4', edgecolor='black', label='$\mu=$0.1')
    plt.bar(r4, mu4, width=bar_width, color='#a3c2f4', edgecolor='black', label='$\mu=$0.01')


    plt.xlabel('Anomaly Type', fontsize=20)
    plt.ylabel('False Negative Rate', fontsize=20)
    plt.xticks([r + bar_width for r in range(len(mu1))], labels, fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim([0.015, 0.022])

    plt.grid(axis='y', linestyle='-')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, fontsize=15, frameon=False, borderaxespad=0,
               columnspacing=0.2, labelspacing=0.1)
    plt.tight_layout()
    plt.savefig("./figure/mu_fnr.pdf")
    plt.show()


def plot_mu_fpr():
    mu1 = [0.03137254901960784, 0.03137254901960784, 0.03137254901960784, 0.03137254901960784]
    mu2 = [0.027450980392156862, 0.027450980392156862, 0.027450980392156862, 0.027450980392156862]
    mu3 = [0.023529411764705882, 0.023529411764705882, 0.023529411764705882, 0.023529411764705882]
    mu4 = [0.0196078431372549, 0.0196078431372549, 0.0196078431372549, 0.0196078431372549]

    # labels = ['False Positive Rate', 'False Negative Rate']
    # labels = ['$\mu=$0.1', '$\mu=$0.1', '$\mu=$0.1', '$\mu=$0.1']
    labels = ['SD', 'MD', 'DM', 'DD']

    bar_width = 0.2

    r1 = np.arange(len(mu1))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    r4 = [x + bar_width for x in r3]

    # plt.bar(r1, y1, width=bar_width,color='#81c9bc',edgecolor='black', label='Loss-guided Dynamic Mask')
    # plt.bar(r2, y2, width=bar_width,color='#e4cc8c',edgecolor='black', label='Top-k Loss Mask')
    # plt.bar(r3, y3, width=bar_width, color='#aee2ff',edgecolor='black', label='Random Mask')
    # plt.bar(r4, y4, width=bar_width,color='#e5e0ff', edgecolor='black', label='w/o Mask')

    plt.bar(r1, mu1, width=bar_width, color='#f39fa1', edgecolor='black', label='$\mu=$100')
    plt.bar(r2, mu2, width=bar_width, color='#ffe699', edgecolor='black', label='$\mu=$10')
    plt.bar(r3, mu3, width=bar_width, color='#c5e0b4', edgecolor='black', label='$\mu=$0.1')
    plt.bar(r4, mu4, width=bar_width, color='#a3c2f4', edgecolor='black', label='$\mu=$0.01')

    plt.xlabel('Anomaly Type', fontsize=20)
    plt.ylabel('False Positive Rate', fontsize=20)
    plt.xticks([r + bar_width for r in range(len(mu1))], labels, fontsize=20)
    plt.yticks(fontsize=20)
    # plt.ylim([0.982, 1])
    plt.grid(axis='y', linestyle='-')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, fontsize=15, frameon=False, borderaxespad=0,
               columnspacing=0.2, labelspacing=0.1)
    plt.tight_layout()
    plt.savefig("./figure/mu_fpr.pdf")
    plt.show()


def plot_mu():
    mu1 = [0.03137254901960784, 0.01764]
    mu2 = [0.027450980392156862, 0.01964]
    mu3 = [0.023529411764705882, 0.02164]
    mu4 = [0.0196078431372549, 0.02164]

    # labels = ['False Positive Rate', 'False Negative Rate']
    labels = ['FPR', 'FNR']

    bar_width = 0.2
    r1 = np.arange(len(mu1))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    r4 = [x + bar_width for x in r3]

    plt.bar(r1, mu1, width=bar_width, color='#f39fa1', edgecolor='black', label='$\mu=$100')
    plt.bar(r2, mu2, width=bar_width, color='#ffe699', edgecolor='black', label='$\mu=$10')
    plt.bar(r3, mu3, width=bar_width, color='#c5e0b4', edgecolor='black', label='$\mu=$0.1')
    plt.bar(r4, mu4, width=bar_width, color='#a3c2f4', edgecolor='black', label='$\mu=$0.01')

    plt.xlabel('Value', fontsize=20)
    plt.ylabel('Metrics', fontsize=20)
    plt.xticks([r + bar_width for r in range(len(mu1))], labels, fontsize=20)
    plt.yticks(fontsize=20)
    # plt.ylim([0.982, 1])
    # 添加y轴标签
    plt.grid(axis='y', linestyle='-')
    # 添加图例
    plt.legend(ncol=2, fontsize=15, frameon=False)
    plt.tight_layout()
    # plt.savefig("./plot/mu_performance.pdf")
    # 显示图表
    plt.show()



def plot_mask_ratio_and_step():
    with open("results/final1_para_an.json", "r") as file_para:
        res = json.loads(file_para.read())

    data_dic = {'SD': np.zeros((8, 4, 4)), 'MD': np.zeros((8, 4, 4)), 'DM': np.zeros((8, 4, 4)),
                'DD': np.zeros((8, 4, 4))}
    for item in res:
        attack_type = item['type']
        # print(attack_type)
        ratio = mask_ratio_dic[item['mask_ratio']]
        step = mask_step_dic[item['mask_step']]
        em = embedding_dic[item['embedding']]
        data_dic[attack_type][em][ratio][step] = item['f1_score']
    print(data_dic)

    plot_heat_map(data_dic['SD'][5], 'SD')
    plot_heat_map(data_dic['MD'][5], 'MD')
    plot_heat_map(data_dic['DM'][5], 'DM')
    plot_heat_map(data_dic['DD'][5], 'DD')



def plot_em():
    with open("results/final_para_an.json", "r") as file_para:
        res = json.loads(file_para.read())

    data_dic = {'SD': np.zeros((8, 4, 4)), 'MD': np.zeros((8, 4, 4)), 'DM': np.zeros((8, 4, 4)),
                'DD': np.zeros((8, 4, 4))}
    for item in res:
        attack_type = item['type']
        # print(attack_type)
        ratio = mask_ratio_dic[item['mask_ratio']]
        step = mask_step_dic[item['mask_step']]
        em = embedding_dic[item['embedding']]
        data_dic[attack_type][em][ratio][step] = item['f1_score']
    print(data_dic)

    plot(list(embedding_dic.keys()), data_dic['SD'][:, 2, 2], data_dic['MD'][:, 2, 2], data_dic['DM'][:, 2, 2], data_dic['DD'][:, 2, 2])



if __name__ == "__main__":

    # Figure 7
    # plot_mask_loss_compare()

    # Figure 8(a)
    # plot_mu_fpr()

    # Figure 8(b)
    # plot_mu_fnr()

    # other parameters like learning rate and batch size
    # plot_lr()
    # plot_batch()



    # Figure 9(a)
    # plot_em()

    # Figure 9(b)
    # plot_layer()


    # Figure 14 (a)
    # plot_mask()

    # Figure 14 (b)
    # plot_mask_var()



    pass
