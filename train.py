import argparse
import random
import torch
from torch import optim
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pickle
import torch.nn as nn
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
import os
from SmartGuard import SmartGuard, TimeSeriesDataset
import json
import time
import math

vocab_dic = {"an": 141, "fr": 222, "sp": 234}
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def get_args_parser():
    # python train.py --epochs 10
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    # an: 20
    parser.add_argument('--epochs', default=60, type=int)
    # Model parameters
    parser.add_argument('--model', default='SmartGuard', type=str, metavar='MODEL',
                        help='Name of model to train: GRUAutoencoder/TransformerAutoencoder/SmartGuard')
    parser.add_argument('--dataset', default='sp', type=str, metavar='MODEL',
                        help='Name of dataset to train: an/fr/us/sp')
    parser.add_argument('--mask_strategy', default='loss_guided', type=str, metavar='MODEL',
                        help='Mask strategy:random/top_k_loss/loss_guided')
    parser.add_argument('--mask_ratio', default=0.2, type=float)
    parser.add_argument('--mask_step', default=4, type=int)
    parser.add_argument('--layer', default=2, type=int)
    parser.add_argument('--batch', default=1024, type=int)
    parser.add_argument('--embedding', default=256, type=int)

    return parser


def make_data(data_file, batch_size):
    with open(data_file, 'rb') as file:
        data = pickle.load(file)
    data = np.array(data)
    dataset = TimeSeriesDataset(data, args.embedding)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return data_loader


def train(args):
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model = nn.DataParallel(model)
    model = SmartGuard(vocab_size=vocab_size, d_model=args.embedding, nhead=8, num_layers=args.layer,
                       mask_strategy=args.mask_strategy, mask_ratio=args.mask_ratio, mask_step=args.mask_step)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    criterion_loss = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = args.epochs

    train_loader = make_data(data_file=train_file1, batch_size=batch_size)
    val_loader = make_data(data_file=vld_file, batch_size=batch_size)

    best_val_loss = 1000
    stop_count = 0
    early_stop_count = 100
    # Train loop

    last_loss_vector = {}
    last_number_vector = {}
    res = []
    for epoch in range(num_epochs):
        t1 = time.time()
        total_loss = 0
        total_loss_all = 0

        loss_vector = {}
        number_vector = {}

        for batch in train_loader:
            input_batch, target_batch, duration_emb = batch
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            optimizer.zero_grad()

            if args.model == "SmartGuard":
                outputs, mask = model(input_batch, last_loss_vector, epoch, duration_emb)
            else:
                outputs = model(input_batch)

            outputs = outputs.view(-1, vocab_size)
            target_batch = target_batch.view(-1)
            target_batch = target_batch.to(dtype=torch.long)

            if args.model == "SmartGuard":
                tmp_mask = []
                if args.mask_strategy == "random" or (args.mask_strategy == "top_k_loss" and epoch == 0):

                    mask = mask.float().masked_fill(mask == float('-inf'), float(1.0))
                else:
                    if epoch <= args.mask_step:
                        mask = mask.float().masked_fill(mask == 0, float(1.0))
                    else:
                        mask = mask.float().masked_fill(mask == float('-inf'), float(1.0))

                for m in mask:
                    tmp_mask.extend(m[0])
                tmp_mask = torch.stack(tmp_mask).to(device)
                loss = F.cross_entropy(outputs, target_batch, reduction="none")
                loss_all = F.cross_entropy(outputs, target_batch, reduction="mean")
                loss = tmp_mask * loss
                loss = torch.sum(loss) / torch.sum(tmp_mask)
            else:
                loss = criterion(outputs, target_batch)
                loss_all = loss

            loss_record = criterion_loss(outputs, target_batch)
            # print(target_batch.shape)
            # print(loss_record.shape)
            for (idx, be) in enumerate(target_batch):
                be = be.item()
                if be in number_vector:
                    number_vector[be] += 1
                else:
                    number_vector[be] = 1
                if be in loss_vector:
                    loss_vector[be] += loss_record[idx].item()
                else:
                    loss_vector[be] = loss_record[idx].item()

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_loss_all += loss_all.item()

        tmps = []
        for key in loss_vector.keys():
            loss_vector[key] = loss_vector[key] / number_vector[key]
            tmps.append(loss_vector[key])

        avg_loss = total_loss / len(train_loader)
        avg_loss_all = total_loss_all / len(train_loader)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {avg_loss:.4f} - ALL Loss: {avg_loss_all:.4f}, Mean:{np.mean(tmps)}, Var:{np.var(tmps)}")
        # break

        # model.eval()
        last_loss_vector = loss_vector
        last_number_vector = number_vector

        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_batch, target_batch, duration_emb = batch
                input_batch = input_batch.to(device)
                target_batch = target_batch.to(device)

                if args.model == "SmartGuard":
                    outputs = model.evaluate(input_batch, duration_emb)
                else:
                    outputs = model(input_batch)

                outputs = outputs.view(-1, vocab_size)
                target_batch = target_batch.view(-1)
                target_batch = target_batch.to(dtype=torch.long)

                loss = criterion(outputs, target_batch)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        # vld_losses.append(avg_val_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Validation Loss: {avg_val_loss:.5f}")
        res.append({"Epoch": epoch + 1, "Train Loss": avg_loss, "ALL Loss": avg_loss_all, "Mean": np.mean(tmps),
                    "Var": np.var(tmps)})

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_name)
            stop_count = 0
        else:
            stop_count += 1
            if stop_count >= early_stop_count:
                print("Early stopping after {} steps without improvement.".format(early_stop_count))
                break

        t2 = time.time()

        print(f'{t2 - t1}')

    with open(f"results/{args.model}_{args.mask_strategy}.json", "w") as js_file:
        json.dump(res, js_file)

    print('Finished Training')


def get_behavior_weight():
    # val_loader = make_data(data_file='data/sp_data/deleted_flattened_useful_sp_vld_instance_10.pkl', batch_size=1)
    val_loader = make_data(data_file=vld_file, batch_size=1)
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model = nn.DataParallel(model)
    # model = model.to(device)
    criterion = nn.CrossEntropyLoss(reduction='none')
    # criterion = nn.CrossEntropyLoss()
    model.load_state_dict(torch.load(model_name))

    model.eval()
    loss_dic = {}
    number_dic = {}
    model.to(device)
    for batch in val_loader:
        input_batch, target_batch, duration_emb = batch

        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)

        if args.model == "SmartGuard":
            outputs = model.evaluate(input_batch, duration_emb)
        else:
            outputs = model(input_batch)

        outputs = outputs.view(-1, vocab_size)
        target_batch = target_batch.view(-1)
        target_batch = target_batch.to(dtype=torch.long)

        loss = criterion(outputs, target_batch)
        for item in input_batch:
            item = item[3:40:4]
            for (idx, behavior) in enumerate(item):
                behavior = behavior.item()
                if behavior in number_dic:
                    number_dic[behavior] += 1
                    loss_dic[behavior] += loss[idx].item()
                else:
                    number_dic[behavior] = 1
                    loss_dic[behavior] = loss[idx].item()

    res = []
    loss_vec = []
    for key in loss_dic.keys():
        loss_dic[key] = loss_dic[key] / number_dic[key]
        loss_vec.append(loss_dic[key])
        res.append((number_dic[key], loss_dic[key], key))
    # print(loss_dic)
    # print(number_dic)
    res = sorted(res, reverse=True)
    # print(res)
    # print(loss_vec)
    # print(len(loss_vec))
    weights = {}
    mean_value = np.mean(np.array(loss_vec))
    var_value = np.var(np.array(loss_vec))

    mu = 0.01

    normal_behaviors = []
    for key in loss_dic.keys():
        weight = sigmoid(-relu(loss_dic[key] - mean_value) / (mu * math.sqrt(var_value)))
        weights[key] = weight

    print(weights)
    print(normal_behaviors)
    return weights


def relu(x):
    return np.maximum(0, x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def find_threshold(percentage=80):
    val_loader = make_data(data_file=vld_file, batch_size=args.batch)
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss(reduction='none')
    # criterion = nn.CrossEntropyLoss()
    model.load_state_dict(torch.load(model_name))
    model.eval()  # 设置模型为评估模式

    losses = []
    model.to(device)
    total_loss = 0
    for batch in val_loader:
        if args.model == "SmartGuard":
            input_batch, target_batch, duration_emb = batch
        else:
            input_batch, target_batch = batch
        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)

        if args.model == "SmartGuard":
            outputs = model.evaluate(input_batch, duration_emb)
        else:
            outputs = model(input_batch)

        outputs = outputs.view(-1, vocab_size)
        target_batch = target_batch.view(-1)
        target_batch = target_batch.to(dtype=torch.long)

        loss = criterion(outputs, target_batch)
        loss = loss.view(-1, 10)
        loss = loss.mean(dim=1)
        losses.extend(loss.cpu().detach().numpy())
        total_loss += loss.sum()

        # losses.append(loss.item())
        # total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    print(f"Avg Loss (Validation Dataset): {avg_loss:.4f}")
    # print(len(losses), len(val_loader))
    threshold = np.percentile(losses, percentage)
    print(f"Percentage:{percentage}% Threshold: {threshold}")
    return threshold


def make_evaluate_data(attack_type):
    file_list = attacks_dic[args.dataset][attack_type]
    X_test_e = []
    for attack_file in file_list:
        with open(f"data2/{args.dataset}_data/attack/labeled_{args.dataset}_{attack_file}.pkl", 'rb') as file1:
            X_test_e += pickle.load(file1)
    for i in range(len(X_test_e)):
        X_test_e[i] = X_test_e[i][0]

    with open(test_file2, 'rb') as file2:
        X_test_r = pickle.load(file2)

    return X_test_r, X_test_e


def evaluate(threshold, weights, attack_type):
    X_test_r, X_test_e = make_evaluate_data(attack_type)
    # X_train = X_trn_r + X_trn_e
    input_test = X_test_r + X_test_e
    labels = [0] * len(X_test_r) + [1] * len(X_test_e)

    model.to(device)
    input_test = np.array(input_test)

    test_dataset = TimeSeriesDataset(input_test, args.embedding)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    criterion = nn.CrossEntropyLoss()
    criterion_loss = nn.CrossEntropyLoss(reduction='none')

    model.load_state_dict(torch.load(model_name))
    model.eval()
    losses = []
    predictions = []

    total_loss = 0
    for batch in test_loader:
        input_batch, target_batch, duration_emb = batch
        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)

        if args.model == "SmartGuard":
            outputs = model.evaluate(input_batch, duration_emb)
        else:
            outputs = model(input_batch)

        outputs = outputs.view(-1, vocab_size)
        target_batch = target_batch.view(-1)
        target_batch = target_batch.to(dtype=torch.long)

        loss = criterion(outputs, target_batch)

        loss_new = criterion_loss(outputs, target_batch)

        for item in input_batch:
            item = item[3:40:4]
            tmp_weight = []
            for (i, be) in enumerate(item):
                if be.item() not in weights:
                    # tmp_weight.append(0.1)
                    tmp_weight.append(0.5)
                else:
                    tmp_weight.append(weights[be.item()])
            for (i, be) in enumerate(item):
                loss_new[i] = loss_new[i] * (tmp_weight[i] / sum(tmp_weight))

        if args.model == "SmartGuard":
            # if args.model == "xxx":
            loss_new = loss_new.view(-1, 10)
            loss_new = loss_new.mean(dim=1)
            losses.extend(loss_new.cpu().detach().numpy())

        else:
            losses.append(loss.item())
            total_loss += loss.item()

    bad_case = []
    for i in range(len(losses)):
        if losses[i] < threshold:
            predictions.append(0)
        else:
            predictions.append(1)
            if labels[i] != 1:
                bad_case.append((input_test[i][3:40:4], losses[i]))

    # 计算混淆矩阵
    cm = confusion_matrix(y_true=labels, y_pred=predictions)
    TN, FP, FN, TP = cm.ravel()
    FPR = FP / (FP + TN)
    FNR = FN / (FN + TP)
    recall = recall_score(y_pred=predictions, y_true=labels)
    precision = precision_score(y_pred=predictions, y_true=labels)
    accuracy = accuracy_score(y_pred=predictions, y_true=labels)
    f1 = f1_score(y_pred=predictions, y_true=labels)

    res = {"dataset": args.dataset, "type": attack_type, "TP": int(TP), "TN": int(TN), "FP": int(FP), "FN": int(FN),
           "FPR": FPR, "FNR": FNR, "recall": recall,
           "precision": precision, "accuracy": accuracy, "f1_score": f1}
    return res


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    print(args)
    vocab_size = vocab_dic[args.dataset]
    train_file1 = f"data/{args.dataset}_data/{args.dataset}_trn_instance_10.pkl"
    train_file2 = f"data/{args.dataset}_data/{args.dataset}_add_trn.pkl"
    vld_file = f"data/{args.dataset}_data/{args.dataset}_vld_instance_10.pkl"
    test_file2 = f"data/{args.dataset}_data/{args.dataset}_test_instance_10.pkl"
    # test_file1 = f"data/{args.dataset}_data/attack/labeled_{args.dataset}_television_attack.pkl"
    batch_size = args.batch
    setup_seed(2023)

    attacks_dic = {
        "an": {"SD": ["light_attack", "camera_attack", "television_attack"],
               "MD": ["smartlock_attack1", "smartlock_attack2"],
               "DM": ["airconditioner_attack", "blind_attack"],
               "DD": ["heater_attack", "bathheater_attack"]},
        "fr": {"SD": ["light_attack", "camera_attack", "television_attack"],
               "MD": ["smartlock_attack1", "smartlock_attack2"],
               "DM": ["airconditioner_attack", "blind_attack"],
               "DD": ["microwave_attack", "oven_attack"]},
        "sp": {"SD": ["light_attack", "camera_attack", "television_attack"],
               "MD": ["smartlock_attack1", "smartlock_attack2"],
               "DM": ["airconditioner_attack", "blind_attack", "watervalve_attack"],
               "DD": ["microwave_attack"]},
        "us": {"SD": ["light_attack", "camera_attack", "television_attack"],
               "MD": ["smartlock_attack1", "smartlock_attack2"],
               "DM": ["airconditioner_attack", "blind_attack", "watervalve_attack"],
               "DD": ["microwave_attack"]},
    }

    model_name = f"saved_model/final1_{args.model}_True_{args.LDMS}_{args.mask_strategy}_{args.mask_ratio}_{args.mask_step}_{args.layer}_{args.embedding}_{args.batch}_{args.dataset}.pth"
    model = SmartGuard(vocab_size=vocab_size, d_model=args.embedding, nhead=8,
                       num_layers=args.layer,
                       mask_strategy=args.mask_strategy, mask_ratio=args.mask_ratio,
                       mask_step=args.mask_step)

    train(args)
