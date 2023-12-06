# -*- coding: utf-8 -*-
import numpy as np
import math

"""
Metrics
"""


def hit_rate(pred, label, lab_len, top_n=3):
    """
    calculate Hit-Rate@k (HR@k)
    """
    eval_num = min(top_n, lab_len)
    hit_num = len(set(pred[:eval_num]) & set(label[:eval_num]))
    hit_rate = hit_num / eval_num
    return hit_rate


def kendall_rank_correlation(pred, label, label_len):
    """
    caculate  kendall rank correlation (KRC), note that label set is a subset of pred set
    """
    def is_concordant(i, j):
        return 1 if (label_order[i] < label_order[j] and pred_order[i] < pred_order[j]) or (
                label_order[i] > label_order[j] and pred_order[i] > pred_order[j]) else 0

    if label_len == 1: return 1

    label = label[:label_len]
    not_in_label = set(pred) - set(label)# 0
    # get order dict
    pred_order = {d: idx for idx, d in enumerate(pred)}
    label_order = {d: idx for idx, d in enumerate(label)}
    for o in not_in_label:
        label_order[o] = len(label)

    n = len(label)
    # compare list 1: compare items between labels
    lst1 = [(label[i], label[j]) for i in range(n) for j in range(i + 1, n)]
    # compare list 2: compare items between label and pred
    lst2 = [(i, j) for i in label for j in not_in_label]

    try:
        hit_lst = [is_concordant(i, j) for i, j in (lst1 + lst2)]
    except:
        print('[warning]: wrong in calculate KRC')
        return float(1)

    hit = sum(hit_lst)
    not_hit = len(hit_lst) - hit
    result = (hit - not_hit) / (len(lst1) + len(lst2))
    return result


def _sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


def idx_weight(i, mode='linear'):
    if mode == 'linear': return 1 / (i + 1)
    if mode == 'exp': return math.exp(-i)
    if mode == 'sigmoid': return _sigmoid(5 - i)  # 5 means we focuse on the top 5
    if mode == 'no_weight': return 1
    if mode == 'log': return 1 / math.log(2 + i)  # i is start from 0


def route_acc(pred, label, top_n):
    """
    calculate ACC@k
    """
    assert set(label).issubset(set(pred)), f"error in prediction:{pred}, label:{label}"
    eval_num = min(top_n, len(label))
    pred = pred[:eval_num]
    if not isinstance(pred, list): pred = pred.tolist()
    if not isinstance(label, list): label = label.tolist()
    for i in range(eval_num):# which means the sub route should be totally correct.
        if not pred[i] == label[i]: return 0
    return 1


def location_deviation(pred, label, label_len, mode='square'):
    """
    calculate LSD / LMD
    mode:
       'square', The Location Square Deviation (LSD)
        else:    The Location Mean Deviation (LMD)
    """

    label = label[:label_len]

    n = len(label)
    # get the location in list 1
    idx_1 = [idx for idx, x in enumerate(label)]
    # get the location in list 2
    for i in range(len(label)):
        if label[i] not in pred:
            print(pred)
            print(label)
    idx_2 = [pred.index(x) for x in label]

    # caculate the distance
    idx_diff = [math.fabs(i - j) for i, j in zip(idx_1, idx_2)]
    weights = [idx_weight(idx, 'no_weight') for idx in idx_1]

    result = list(map(lambda x: x ** 2, idx_diff)) if mode == 'square' else idx_diff
    return sum([diff * w for diff, w in zip(result, weights)]) / n

def edit_distance(pred, label):
    """
    calculate edit distance (ED)
    """
    import edit_distance
    assert set(label).issubset(set(pred)), "error in prediction"
    # Focus on the items in the label
    if not isinstance(pred, list): pred = pred.tolist()
    if not isinstance(label, list): label = label.tolist()
    try:
         pred = [x for x in pred if x in label]
         ed = edit_distance.SequenceMatcher(pred, label).distance()
    except:
           print('pred in function:', pred, f'type of pred: {type(pred)}')
           print('label in function:', label, f'type label:{type(label)}')
    return ed

def calc_rmse(pred, label):
    valid_pred = pred[:len(label)]
    return np.sqrt(np.sum(((np.array(valid_pred) - np.array(label))**2/len(label))))

def calc_mae(pred, label):
    valid_pred = pred[:len(label)]
    return np.sum(np.abs(np.array(valid_pred) - np.array(label)))/len(label)

def acc_eta(pred, label, top_n):
    valid_pred = pred[:len(label)]
    return len((np.abs(np.array(valid_pred) - np.array(label)) <= top_n).nonzero()[0]) /len(label)

def calc_mape(pred, label):
    valid_pred = pred[:len(label)]
    return np.sum(np.abs(np.array(valid_pred) - np.array(label))/np.array(label))/len(label)

def calc_picp(pred, sigma, label):
    upper = np.array(pred) + np.array(sigma)
    lower = np.array(pred)  - np.array(sigma)
    count = np.sum((label > lower) & (label < upper))

    return count / len(pred)
import torch
def calc_mis(pred, sigma, label): # 越小越好
    pred = torch.tensor(pred).float()
    sigma = torch.tensor(sigma).float()
    label = torch.tensor(label).float()
    pred_upper = pred + sigma
    pred_lower = pred - sigma
    l0 = torch.mean(torch.abs(pred - label))
    l1 = torch.mean((torch.max(pred_upper - pred_lower, torch.tensor([0.]))))
    l2 = torch.mean(torch.max(pred_lower - label, torch.tensor([0.]))) * 40
    l3 = torch.mean(torch.max(label - pred_upper, torch.tensor([0.]))) * 40

    return (l0 + l1 + l2 + l3).item()

def calc_interval(pred, sigma, label):
    pred = torch.tensor(pred)
    sigma = torch.tensor(sigma)
    label = torch.tensor(label)
    pred_upper = pred + sigma
    pred_lower = pred - sigma

    l1 = torch.mean((torch.max(pred_upper - pred_lower, torch.tensor([0.]))))

    return (l1).item()
from typing import Dict

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Metric(object):
    def __init__(
            self,
            length_range,
            max_seq_len = 25,
    ):
        self.max_seq_len = max_seq_len
        self.hr = [AverageMeter() for _ in range(3)]
        self.lsd = AverageMeter()
        self.krc = AverageMeter()
        self.lmd = AverageMeter()
        self.ed = AverageMeter() #edit distance
        self.acc = [AverageMeter() for _ in range(3)]
        self.mae = AverageMeter()
        self.rmse = AverageMeter()
        self.mape = AverageMeter()
        self.picp = AverageMeter()
        self.mis = AverageMeter()
        self.interval = AverageMeter()
        self.len_range = length_range
        self.acc_eta = [AverageMeter() for _ in [10, 20, 30, 40, 50, 60]]
        self.acc_eta_list = [10, 20, 30, 40, 50, 60]


    def route_eta_filter_len(self, prediction, label, label_len, eta_pred, eta_sigma, eta_label, eta_label_len):
        """
        filter the input data,  only evalution the data within len_range
        """
        pred_f = []
        label_f = []
        label_len_f = []
        # input_len_f = []
        eta_pred_f = []
        eta_label_f = []
        eta_label_len_f = []
        eta_sigma_f = []
        for i in range(len(label_len)):
            if self.len_range[0] <= label_len[i] <= self.len_range[1]: # label_len, filter by newly arrival
                pred_f.append(prediction[i])
                label_f.append(label[i])
                label_len_f.append(label_len[i])
                eta_pred_f.append(eta_pred[i])
                eta_sigma_f.append(eta_sigma[i])
                eta_label_f.append(eta_label[i])
                eta_label_len_f.append(eta_label_len[i])
        return pred_f, label_f, label_len_f, eta_pred_f, eta_sigma_f, eta_label_f, eta_label_len_f

    def get_route_prediction(self, prediction, label, label_len):
        pred_f = []
        label_f = []
        label_len_f = []
        for i in range(len(label_len)):
            if label_len[i] != 0:  # pred_len
                pred_f.append(prediction[i])
                label_f.append(label[i])
                label_len_f.append(label_len[i])

        return pred_f, label_f, label_len_f

    def update_route_eta(self, prediction, label, label_len, eta_pred, eta_sigma, eta_label, eta_label_len) :
        def tensor2lst(x):
            try:
                return x.cpu().numpy().tolist()
            except:
                return x

        prediction, label, label_len, eta_pred, eta_sigma, eta_label, eta_label_len = [tensor2lst(x) for x in [prediction, label, label_len, eta_pred, eta_sigma, eta_label, eta_label_len]]

        # process the prediction
        prediction, label, label_len, eta_pred, eta_sigma, eta_label, eta_label_len = self.route_eta_filter_len(prediction, label, label_len, eta_pred, eta_sigma, eta_label, eta_label_len)
        prediction, label, label_len = self.get_route_prediction(prediction, label, label_len)

        pred = []
        for p in prediction:
            input = set([x for x in p if x < len(prediction[0]) - 1])
            tmp = list(filter(lambda pi: pi in input, p))
            pred.append(tmp)

        batch_size = len(pred)

        for n in range(3):
            hr_n = np.array([hit_rate(pre, lab, lab_len, n + 1) for pre, lab, lab_len in zip(pred, label, label_len)]).mean()
            self.hr[n].update(hr_n, batch_size)

        krc = np.array([kendall_rank_correlation(pre, lab, lab_len) for pre, lab, lab_len in zip(pred, label, label_len)]).mean()
        self.krc.update(krc, batch_size)

        lsd = np.array([location_deviation(pre, lab, lab_len, 'square') for pre, lab, lab_len in zip(pred, label, label_len)]).mean()
        self.lsd.update(lsd, batch_size)

        lmd = np.array([location_deviation(pre, lab, lab_len, 'mean') for pre, lab, lab_len in zip(pred, label, label_len)]).mean()
        self.lmd.update(lmd, batch_size)

        ed = np.array([edit_distance(pre, lab[:lab_len]) for pre, lab, lab_len in zip(pred, label, label_len)]).mean()
        self.ed.update(ed, batch_size)

        for n in range(3):
            acc_n = np.array([route_acc(pre, lab[:lab_len], n + 1) for pre, lab, lab_len in zip(pred, label, label_len)]).mean()
            self.acc[n].update(acc_n, batch_size)

        mae = np.sum(np.array([calc_mae(pre, lab[:lab_len]) for pre, lab, lab_len in zip(eta_pred, eta_label, label_len)]) * label_len) / np.sum(label_len)
        self.mae.update(mae, batch_size)
        rmse = np.sum(np.array([calc_rmse(pre, lab[:lab_len]) for pre, lab, lab_len in zip(eta_pred, eta_label, label_len)]) * label_len) / np.sum(label_len)
        self.rmse.update(rmse, batch_size)
        mape = np.sum(np.array([calc_mape(pre, lab[:lab_len]) for pre, lab, lab_len in zip(eta_pred, eta_label, label_len)]) * label_len) / np.sum(label_len)
        self.mape.update(mape, batch_size)

        for n in range(len(self.acc_eta_list)):
            acc_eta_n = np.sum(np.array([acc_eta(pre, lab[:lab_len], self.acc_eta_list[n]) for pre, lab, lab_len in zip(eta_pred, eta_label, label_len)]) * label_len) / np.sum(label_len)
            self.acc_eta[n].update(acc_eta_n, batch_size)

        picp = np.sum(np.array([calc_picp(pre[:lab_len], sigma[:lab_len], lab[:lab_len]) for pre, lab, sigma, lab_len in zip(eta_pred, eta_label, eta_sigma, label_len)]) * label_len) / np.sum(label_len)
        self.picp.update(picp, batch_size)

        mis = np.sum(np.array([calc_mis(pre[:lab_len], sigma[:lab_len], lab[:lab_len]) for pre, lab, sigma, lab_len in zip(eta_pred, eta_label, eta_sigma, label_len)]) * label_len) /  np.sum(label_len)
        self.mis.update(mis, batch_size)

        interval = np.sum(np.array([calc_interval(pre[:lab_len], sigma[:lab_len], lab[:lab_len]) for pre, lab, sigma, lab_len in zip(eta_pred, eta_label, eta_sigma, label_len)])* label_len) / np.sum(label_len)
        self.interval.update(interval, batch_size)


    def route_eta_to_dict(self) -> Dict:
        result = {f'hr@{i + 1}': self.hr[i].avg for i in range(3)}
        result.update({f'acc@{i + 1}': self.acc[i].avg for i in range(3)})
        result.update({'lsd': self.lsd.avg, 'lmd': self.lmd.avg, 'krc': self.krc.avg, 'ed': self.ed.avg})
        result.update({'mape': self.mape.avg, 'rmse': self.rmse.avg, 'mae': self.mae.avg, 'picp': self.picp.avg, 'mis': self.mis.avg, 'interval': self.interval.avg})
        result.update({f'acc_eta@{(i+1)*10}': self.acc_eta[i].avg for i in range(len(self.acc_eta_list))})
        return result

    def route_eta_to_str(self):
        hr = [round(x.avg, 3) for x in self.hr]
        acc = [round(x.avg, 3) for x in self.acc]
        krc = round(self.krc.avg, 3)
        lsd = round(self.lsd.avg, 3)
        ed = round(self.ed.avg, 3)
        rmse = round(self.rmse.avg, 3)
        mae = round(self.mae.avg, 3)
        mape = round(self.mape.avg, 3)
        picp = round(self.picp.avg, 5)
        mis = round(self.mis.avg, 5)
        interval = round(self.interval.avg, 5)
        acc_eta = [round(x.avg, 3) for x in self.acc_eta]
        return f'krc:{krc} | lsd:{lsd} | ed:{ed} | hr@1:{hr[0]} | hr@2:{hr[1]} | hr@3:{hr[2]} | acc@1:{acc[0]} |' \
               f' acc@2:{acc[1]} | acc@3:{acc[2]}|mape:{mape} | rmse:{rmse} | mae: {mae} |acc_eta@20:{acc_eta[1]}|acc_eta@30:{acc_eta[2]}|picp:{picp}|mis:{mis}|interval:{interval}'
