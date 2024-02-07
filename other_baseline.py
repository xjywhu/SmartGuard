from sklearn.ensemble import IsolationForest
from sklearn import svm
from sklearn.neighbors import LocalOutlierFactor
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import GaussianNB
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
import argparse

vocab_dic = {"an": 141, "fr": 222, "us": 268, "sp": 234}


class MarkovChain():
    def __init__(self, state_number):
        self.transition_matrix = None
        self.states = None
        self.state_number = state_number

    def fit(self, sequences):
        # 从数据中提取所有独特的状态
        # self.states = set(x for sequence in sequences for x in sequence)
        self.states = set(x for x in range(self.state_number))

        # 初始化转移矩阵
        state_index = {state: i for i, state in enumerate(self.states)}
        n_states = len(self.states)
        # transition_matrix = np.zeros((n_states, n_states))
        transition_matrix = np.ones((n_states, n_states)) * 1e-10

        # 计算转移矩阵
        for sequence in sequences:
            for i in range(len(sequence) - 1):
                state_from, state_to = sequence[i], sequence[i + 1]
                # print(state_from, state_to)
                # if state_to == 228:
                #     print(state_to)
                # print(sequence)
                transition_matrix[state_index[state_from]][state_index[state_to]] += 1

        # 归一化转移矩阵
        transition_matrix /= transition_matrix.sum(axis=1, keepdims=True)

        self.transition_matrix = transition_matrix

    def predict_sequence_probability(self, sequence):
        state_index = {state: i for i, state in enumerate(self.states)}
        probability = 1.0
        for i in range(len(sequence) - 1):
            probability *= self.transition_matrix[state_index[sequence[i]]][state_index[sequence[i + 1]]]

        return probability


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    # Model parameters
    parser.add_argument('--model', default='OCSVM', type=str, metavar='MODEL',
                        help='Name of model to train: GMM/NB/LocalOutlierFactor/IsolationForest/Autoencoder/MC/OCSVM')
    parser.add_argument('--dataset', default='fr', type=str, metavar='MODEL',
                        help='Name of dataset to train: an/fr/us/sp')
    parser.add_argument('--attack', default='SD', type=str, metavar='type',
                        help='Name of dataset to train: SD/MD/DM/DD')
    return parser


def make_data():
    with open(train_file1, 'rb') as file1:
        X_trn_r1 = pickle.load(file1)

    # with open(train_file2, 'rb') as file4:
    #     X_trn_r2 = pickle.load(file4)
    # with open('us_television_attack.pkl', 'rb') as file3:
    #     X_e = pickle.load(file3)
    if args.attack == 'SD':
        with open(f"data/{args.dataset}_data/attack/{args.dataset}_light_attack.pkl", 'rb') as file3:
            X_e1 = pickle.load(file3)
        with open(f"data/{args.dataset}_data/attack/{args.dataset}_camera_attack.pkl", 'rb') as file3:
            X_e2 = pickle.load(file3)
        with open(f"data/{args.dataset}_data/attack/{args.dataset}_television_attack.pkl", 'rb') as file3:
            X_e3 = pickle.load(file3)
        X_test_e = X_e1 + X_e2 + X_e3
        # X_test_e = X_e1 + X_e2
        # X_test_e = X_e1
    elif args.attack == 'MD':
        with open(f"data/{args.dataset}_data/attack/{args.dataset}_smartlock_attack1.pkl", 'rb') as file3:
            X_e1 = pickle.load(file3)
        with open(f"data/{args.dataset}_data/attack/{args.dataset}_smartlock_attack2.pkl", 'rb') as file3:
            X_e2 = pickle.load(file3)
        X_test_e = X_e1 + X_e2
    elif args.attack == 'DM':
        if args.dataset != "an":
            with open(f"data/{args.dataset}_data/attack/{args.dataset}_airconditioner_attack.pkl", 'rb') as file3:
                X_e1 = pickle.load(file3)

        with open(f"data/{args.dataset}_data/attack/{args.dataset}_blind_attack.pkl", 'rb') as file3:
            X_e2 = pickle.load(file3)

        if args.dataset == "fr":
            X_test_e = X_e1 + X_e2
        elif args.dataset == "an":
            X_test_e = X_e2
        else:
            with open(f"data/{args.dataset}_data/attack/{args.dataset}_watervalve_attack.pkl", 'rb') as file3:
                X_e3 = pickle.load(file3)
            X_test_e = X_e1 + X_e2 + X_e3
    else:
        if args.dataset == "an":
            with open(f"data/{args.dataset}_data/attack/{args.dataset}_bathheater_attack.pkl", 'rb') as file3:
                X_e1 = pickle.load(file3)
        else:
            with open(f"data/{args.dataset}_data/attack/{args.dataset}_microwave_attack.pkl", 'rb') as file3:
                X_e1 = pickle.load(file3)
        X_test_e = X_e1

    # with open(f'{args.dataset}_add_blind_attack.pkl', 'rb') as file3:
    #     X_e2 = pickle.load(file3)
    with open(test_file2, 'rb') as file2:
        X_test_r = pickle.load(file2)

    return X_trn_r1, X_test_r, X_test_e


def count_cm(labels, y_pred_test):
    cm = confusion_matrix(y_true=labels, y_pred=y_pred_test)
    TN, FP, FN, TP = cm.ravel()

    FPR = FP / (FP + TN)
    FNR = FN / (FN + TP)
    recall = recall_score(y_pred=y_pred_test, y_true=labels)
    precision = precision_score(y_pred=y_pred_test, y_true=labels)
    accuracy = accuracy_score(y_pred=y_pred_test, y_true=labels)
    f1 = f1_score(y_pred=y_pred_test, y_true=labels)

    res = {"dataset": args.dataset, "type": args.attack, "TP": int(TP), "TN": int(TN), "FP": int(FP), "FN": int(FN),
           "FPR": FPR, "FNR": FNR, "recall": recall,
           "precision": precision, "accuracy": accuracy, "f1_score": f1}

    return res


def train(args):
    labels = []
    y_pred_test = []
    if args.model == "GMM":
        X_train, X_test_r, X_test_e = make_data()
        # X_train = X_trn_r + X_trn_e
        X_test = X_test_r + X_test_e
        labels = [0] * len(X_test_r) + [1] * len(X_test_e)

        model = GaussianMixture(n_components=2, random_state=2023)
        X_test = np.array(X_test)
        X_test = X_test[:, 3:40:4]
        y_pred_test = model.fit_predict(X_test)
        # print("测试集异常值预测:", y_pred_test)


    elif args.model == "NB":
        X_train, X_test_r, X_test_e = make_data()
        # X_train = X_trn_r + X_trn_e
        X_test = X_test_r + X_test_e
        labels = [0] * len(X_test_r) + [1] * len(X_test_e)
        model = GaussianNB()
        X_test = np.array(X_test)
        X_test = X_test[:, 3:40:4]

        model.fit(X_test, labels)
        # y_pred_test = model.predict(X_test)
        y_pred_test = model.predict(X_test)
        # print(y_pred_test)

        # print("测试集异常值预测:", y_pred_test)


    elif args.model == "LocalOutlierFactor":
        X_train, X_test_r, X_test_e = make_data()
        X_test = X_test_r + X_test_e
        labels = [0] * len(X_test_r) + [1] * len(X_test_e)
        # sp:(n_neighbors=2)
        model = LocalOutlierFactor(n_neighbors=10, contamination='auto')
        X_test = np.array(X_test)
        X_test = X_test[:, 3:40:4]
        y_pred_test = model.fit_predict(X_test)
        y_pred_test = np.array(y_pred_test)
        y_pred_test[np.where(y_pred_test == 1)] = 0
        y_pred_test[np.where(y_pred_test == -1)] = 1
        # print("测试集异常值预测:", y_pred_test)


    elif args.model == "IsolationForest":
        X_train, X_test_r, X_test_e = make_data()
        X_test = X_test_r + X_test_e
        labels = [0] * len(X_test_r) + [1] * len(X_test_e)

        X_test = np.array(X_test)
        X_test = X_test[:, 3:40:4]
        # 将一些数据点标记为异常值
        model = IsolationForest(contamination='auto', random_state=2023)
        # 预测异常值
        y_pred_test = model.fit_predict(X_test)
        y_pred_test = np.array(y_pred_test)
        y_pred_test[np.where(y_pred_test == 1)] = 0
        y_pred_test[np.where(y_pred_test == -1)] = 1

        # print(y_pred_test)
        # 输出结果
        # print("测试集异常值预测:", y_pred_test)
        # count_cm(labels, y_pred_test)

    elif args.model == "MC":
        X_train, X_test_r, X_test_e = make_data()
        labels = [1] * len(X_test_r) + [-1] * len(X_test_e)
        X_train = np.array(X_train)
        X_train = X_train[:, 3:40:4]

        X_test = X_test_r + X_test_e
        X_test = np.array(X_test)
        X_test = X_test[:, 3:40:4]

        model = MarkovChain(state_number=vocab_dic[args.dataset])
        model.fit(X_train)

        # 检测异常
        # threshold = 0.00001  # 定义阈值
        # sp: 0.2 us:
        threshold = 0.2
        y_pred_test = []
        for sequence in X_test:
            probability = model.predict_sequence_probability(sequence)
            # print(probability)
            if probability < threshold:
                y_pred_test.append(1)
            else:
                y_pred_test.append(-1)

        # count_cm(labels, y_pred_test)

    elif args.model == "OCSVM":
        X_train, X_test_r, X_test_e = make_data()
        X_test = X_test_r + X_test_e
        labels = [0] * len(X_test_r) + [1] * len(X_test_e)
        X_test = np.array(X_test)
        X_test = X_test[:, 3:40:4]

        # gamma=0.5
        model = svm.OneClassSVM(nu=0.5, kernel="rbf", gamma=0.5)
        y_pred_test = model.fit_predict(X_test)
        y_pred_test[np.where(y_pred_test == 1)] = 0
        y_pred_test[np.where(y_pred_test == -1)] = 1
        # print("测试集异常值预测:", y_pred_test)

    return count_cm(labels, y_pred_test)


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()

    attacks_dic = {
        "SD": ["light_attack", "camera_attack", "television_attack"],
        "MD": ["smartlock_attack1", "smartlock_attack2"],
        "DM": ["airconditioner_attack", "blind_attack", "watervalve_attack"],
        "DD": ["microwave_attack"]
    }

    models = ["GMM", "NB", "LocalOutlierFactor", "IsolationForest", "MC", "OCSVM"]
    datasets = ["an", "fr", "sp", "us"]
    results = []

    for model in models:
        args.model = model
        for dataset in datasets:
            args.dataset = dataset
            train_file1 = f"data/{args.dataset}_data/deleted_flattened_useful_{args.dataset}_trn_instance_10.pkl"
            train_file2 = f"data/{args.dataset}_data/{args.dataset}_add_trn.pkl"
            vld_file = f"data/{args.dataset}_data/deleted_flattened_useful_{args.dataset}_vld_instance_10.pkl"
            # test_file1 = f"data/{args.dataset}_data/attack/{args.dataset}_television_attack.pkl"
            test_file2 = f"data/{args.dataset}_data/deleted_flattened_useful_{args.dataset}_test_instance_10.pkl"

            for attack_type in ["SD", "MD", "DM", "DD"]:
                args.attack = attack_type
                res = train(args)
                tmp = {model: res}
                results.append(tmp)
                print(tmp)
    # with open("results/res.json", "w") as file_res:
    #     file_res.write(json.dumps(results))
    # args.model = model
    # print(args)
    # import time
    #
    # t1 = time.time()
    # train(args)
    # t2 = time.time()
    # print('time:', t2 - t1)
