import json
import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


def plot_mask_loss():
    with open("results/loss_dic.json") as data_file:
        data_dic = json.loads(data_file.read())
    plt.rcParams['figure.figsize'] = 13, 6
    y1 = data_dic['num']

    y2 = data_dic['no_mask']
    y3 = data_dic['random']
    # y3 = data_dic['autoencoder']
    y4 = data_dic['top_k_loss']
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


    ax2.plot(x[0:76], y2[0:76], color='#c00001', marker='o', markersize=8, label='w/o Mask')
    # ax2.plot(x, y3, marker='o', markersize=8, label='Random Mask')
    # ax2.plot(x, y4, marker='o', markersize=8, label='Tok-k Loss Mask')
    # ax2.plot(x, y5, marker='o', markersize=8, label='Loss-guided Dynamic Mask')
    ax2.set_ylabel('Reconstruction Loss', fontsize=25)


    # legend.get_texts().set_fontsize(20)  # 设置字体大小
    # ax2.axhline(y=0.5, color='gray', linestyle='--', linewidth=4)

    ax2.set_ylim(0, 0.022)
    ax1.set_xlabel('Behavior ID', fontsize=25)

    ax1.tick_params(axis='both', which='major', labelsize=25)

    ax2.tick_params(axis='both', which='major', labelsize=25)

    plt.tight_layout()
    plt.savefig("figure/unbalance.pdf")
    plt.show()


if __name__ == "__main__":
    # Figure 1
    plot_mask_loss()