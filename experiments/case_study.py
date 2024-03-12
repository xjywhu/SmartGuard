import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


def case():
    reconstruction_loss = [8.7670e-02, 5.6048e-02, 5.6894e-02, 6.0945e-02, 4.4229e-02, 4.5406e-01,
                           2.8155e-01, 2 * 6.8724e-02, 5.9778e-02, 1.1152e-02]
    with open("case/attention.pkl", 'rb') as att_file:
        attention_data = pickle.load(att_file).detach().numpy()
    sns.heatmap(attention_data, annot=False, fmt=".1f", cmap="YlGnBu", cbar=True, annot_kws={'size': 14})
    labels = ["b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8", "b9", "b10"]
    #
    plt.xticks(np.arange(len(attention_data)) + 0.5, labels=labels, fontsize=15, rotation=360)
    plt.yticks(np.arange(len(attention_data)) + 0.5, labels=labels, fontsize=15, rotation=360)

    plt.tight_layout()
    plt.savefig("figure/attention.pdf")
    plt.show()

    sns.heatmap(np.array(reconstruction_loss).reshape(10, 1), annot=True, fmt=".4f", cmap="YlGnBu", cbar=True,
                annot_kws={'size': 14}, xticklabels=False)
    plt.yticks(np.arange(len(attention_data)) + 0.5, labels=labels, fontsize=15, rotation=360)
    plt.tight_layout()
    plt.savefig("figure/rec_loss.pdf")
    plt.show()


def time_embedding():
    data_dic = {'Light:switch off': 0,
                'Light:switch on': 1,
                'Light:switch toggle': 2,
                'Dishwasher:refresh refresh': 0,
                'Dishwasher:start': 1,
                'Blind:switchLevel setLevel': 0,
                'Blind:windowShade close': 1,
                'Blind:windowShade open': 2,
                'TV_on': 0,
                'TV_off': 1,
                'TV_light_up': 2,
                'TV_light_down': 3,
                'TV_sound_up': 4,
                'TV_sound_down': 5,
                'heater_on': 0,
                'heater_off': 1,
                'heater_temper_up': 2,
                'heater_temper_down': 3,
                'washing_machine_wash': 0,
                'washing_machine_finish': 1,
                'washing_machine_stop': 2,
                'washing_machine_change_mode': 3}

    # hour
    hours = {
        "blind": [[0.15040039, 0., 0.04173148, 0.2316774, 0.4454955, 0.55749935, 0.5183269, 0.37951186],
                  [0.03292596, 0., 0.00963497, 0.10618293, 0.27207613, 0.4476853, 0.5721823, 0.6210541],
                  [0., 0.20355652, 0.40200678, 0.50270325, 0.48137558, 0.38766864, 0.30042958, 0.26828596]],
        "dishwasher": [[0.07998052, 0., 0.04006187, 0.1770366, 0.34487447, 0.47678345, 0.54393107, 0.56438845],
                       [0., 0.01687032, 0.14314167, 0.33951956, 0.5096524, 0.5606201, 0.46423662, 0.2727566]],
        "light": [[0., 0.22691809, 0.38872787, 0.41169268, 0.3339425, 0.2769386, 0.35086665, 0.56266165],
                  [0.37443292, 0.31056118, 0.13350731, 0., 0.05643303, 0.2957555, 0.5440454, 0.59907097],
                  [0.44993436, 0.48804566, 0.49581295, 0.43734783, 0.3109402, 0.15602069, 0.03522442, 0.]]
    }

    days = {
        "TV": [[0.57523656, 0.42798194, 0.1777102, 0., 0.00958187, 0.18347508, 0.39343122],
               [0.22924362, 0.10893521, 0., 0.05190876, 0.28371271, 0.5616606, 0.701784],
               [0.24780077, 0.32474104, 0.22629659, 0.03011564, 0., 0.3109809, 0.80751795],
               [0.05070464, 0.18278691, 0.3724405, 0.5231999, 0.54009557, 0.42149568, 0.28671613],
               [0.01445827, 0.17795114, 0.4096418, 0.56809664, 0.55157316, 0.3777776, 0.17502916],
               [0.12408666, 0.24435234, 0.31260246, 0.34946987, 0.40337995, 0.48748615, 0.5519844]],
        "Washing Machine": [[0.06323512, 0., 0.12381919, 0.3550047, 0.5377336, 0.556469, 0.42911124],
                            [0.56381845, 0.44378456, 0.21979247, 0.03027785, 0., 0.14213628, 0.35378072],
                            [0.14246021, 0., 0.15788403, 0.3747147, 0.4379065, 0.3662953, 0.36879152],
                            [0.29707396, 0.5702874, 0.61667943, 0.41168222, 0.12869266, 0.00315623, 0.14202787]]
    }
    durations = {
        "Heater": [
            [0.53311414, 0.47387928, 0.4146441, 0.35540947, 0.29617453, 0.23693913, 0.17770429, 0.11846986, 0.05923446,
             0.],
            [0.53311396, 0.4738791, 0.41464415, 0.35540932, 0.2961744, 0.23693949, 0.17770486, 0.11846981, 0.05923521,
             0.],
            [0.53311396, 0.47387907, 0.41464424, 0.3554093, 0.29617444, 0.23693961, 0.17770498, 0.11846973, 0.05923477,
             0.],
            [0.53311384, 0.47387904, 0.41464418, 0.35540938, 0.29617435, 0.23693976, 0.17770469, 0.11847012, 0.0592357,
             0.]],
        "Washing Machine": [
            [0.533114, 0.47387916, 0.4146442, 0.35540932, 0.29617447, 0.23693945, 0.17770438, 0.11846968, 0.05923488,
             0.],
            [0., 0.05923487, 0.11846992, 0.17770462, 0.23693953, 0.2961747, 0.35540956, 0.41464412, 0.4738789,
             0.53311396],
            [0.5331138, 0.47387904, 0.41464406, 0.35540956, 0.29617426, 0.23693961, 0.17770496, 0.11847061, 0.05923532,
             0.],
            [0., 0.05923487, 0.11846972, 0.17770462, 0.2369396, 0.29617435, 0.35540944, 0.41464412, 0.47387916,
             0.5331141]]

    }
    duration_labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    hour_y_labels = ['Blind:close', 'Blind:open', 'Dishwasher:start']
    # hour_y_labels = ['Blind:setLevel', 'Blind:close', 'Blind:open', 'Dishwasher:refresh', 'Dishwasher:start']
    data = [durations['Heater'][0]]

    figsize = (8, 1.2)
    # rotation=270
    plt.figure(figsize=figsize)
    sns.heatmap(data, annot=False, fmt=".1f", cmap="YlGnBu", cbar=True, annot_kws={'size': 14}, linewidths=0.1)
    plt.xticks(np.arange(len(durations['Heater'][0])) + 0.5, labels=duration_labels, fontsize=15, rotation=0)
    # plt.yticks(np.arange(len(hour_y_labels)) + 0.5, labels=hour_y_labels, fontsize=15, rotation=360)
    plt.tight_layout()
    plt.savefig("figure/duration.pdf")
    plt.show()


    # hour
    # day_labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    hour_y_labels = ['Blind:close', 'Blind:open', 'Dishwasher:start']
    # hour_y_labels = ['Blind:setLevel', 'Blind:close', 'Blind:open', 'Dishwasher:refresh', 'Dishwasher:start']
    data = days['TV'][1:3]
    data.append(days['Washing Machine'][0])
    # for item in days['Washing Machine']:
    #     data.append(item)
    figsize = (8, 2.5)
    plt.figure(figsize=figsize)
    sns.heatmap(data, annot=False, fmt=".1f", cmap="YlGnBu", cbar=True, annot_kws={'size': 14}, linewidths=0.1)
    plt.xticks(np.arange(len(days['TV'][0])) + 0.5, labels=day_labels, fontsize=15, rotation=0)
    # plt.yticks(np.arange(len(hour_y_labels)) + 0.5, labels=hour_y_labels, fontsize=15, rotation=360)
    plt.tight_layout()
    plt.savefig("figure/day.pdf")
    plt.show()

    # hour
    hour_labels = ['0-3', '3-6', '6-9', '9-12', '12-15', '15-18', '18-21', '21-24']
    data = hours['blind'][1:]
    data.append(hours['dishwasher'][1])
    figsize = (8, 2.5)
    plt.figure(figsize=figsize)
    sns.heatmap(data, annot=False, fmt=".1f", cmap="YlGnBu", cbar=True, annot_kws={'size': 14}, linewidths=0.1)
    plt.xticks(np.arange(len(hours['blind'][0])) + 0.5, labels=hour_labels, fontsize=15, rotation=0)
    plt.tight_layout()
    plt.savefig("figure/hour.pdf")
    plt.show()
    pass


if __name__ == "__main__":
    case()
    # time_embedding()
