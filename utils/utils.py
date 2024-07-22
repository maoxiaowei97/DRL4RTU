# -*- coding: utf-8 -*-
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tqdm import tqdm

import time
import algorithm.DRL4RTU.model as Agent

def get_workspace():
    """
    get the workspace path
    :return:
    """
    cur_path = os.path.abspath(__file__)
    file = os.path.dirname(cur_path)
    file = os.path.dirname(file)
    return file


ws = get_workspace()


def dir_check(path):
    """
    check weather the dir of the given path exists, if not, then create it
    """
    import os
    dir = path if os.path.isdir(path) else os.path.split(path)[0]
    if not os.path.exists(dir): os.makedirs(dir)



def save2file_(params):
    file_name = ws + f'/output/{params["model"]}.csv'
    # 写表头
    head = [
        # data setting
        'dataset', 'min_task_num', 'max_task_num', 'eval_min', 'eval_max',
        # mdoel parameters
        'model', 'hidden_size', 'rl_ratio', 'target',
        'reward_type',
        # training set
        'num_epoch', 'batch_size', 'lr', 'wd', 'early_stop', 'is_test', 'log_time', 'cred',
        'gated_fusion',
        #route metrics
        'lsd', 'lmd', 'krc', 'hr@1', 'hr@2', 'hr@3', 'ed',
        #time metrics
        'mape', 'mae', 'rmse', 'acc_eta@10', 'acc_eta@20', 'acc_eta@30', 'acc_eta@40', 'acc_eta@50', 'acc_eta@60',
        'picp', 'mis', 'interval'
    ]
    save2file_meta(params, file_name, head)

def whether_stop(metric_lst=[], n=2, mode='maximize'):
    '''
    For fast parameter search, judge wether to stop the training process according to metric score
    n: Stop training for n consecutive times without rising
    mode: maximize / minimize
    '''
    if len(metric_lst) < 1: return False  # at least have 2 results.
    if mode == 'minimize': metric_lst = [-x for x in metric_lst]
    max_v = max(metric_lst)
    max_idx = 0
    for idx, v in enumerate(metric_lst):
        if v == max_v: max_idx = idx
    return max_idx < len(metric_lst) - n



class EarlyStop():

    def __init__(self, mode='maximize', patience=1):
        self.mode = mode
        self.patience = patience
        self.metric_lst = []
        self.stop_flag = False
        self.best_epoch = -1  # the best epoch
        self.is_best_change = False  # whether the best change compare to the last epoch

    def append(self, x):
        self.metric_lst.append(x)
        # update the stop flag
        self.stop_flag = whether_stop(self.metric_lst, self.patience, self.mode)
        # update the best epoch
        best_epoch = self.metric_lst.index(max(self.metric_lst)) if self.mode == 'maximize' else self.metric_lst.index(
            min(self.metric_lst))
        if best_epoch != self.best_epoch:
            self.is_best_change = True
            self.best_epoch = best_epoch  # update the wether best change flag
        else:
            self.is_best_change = False
        return self.is_best_change

    def best_metric(self):
        if len(self.metric_lst) == 0:
            return -1
        else:
            return self.metric_lst[self.best_epoch]

def dict_merge(dict_list=[]):
    dict_ = {}
    for dic in dict_list:
        assert isinstance(dic, dict), "object is not a dict!"
        dict_ = {**dict_, **dic}
    return dict_


def get_dataset_path(params={}):
    dataset = params['dataset']
    file = ws + f'/data/dataset/{dataset}'
    train_path = file + f'/train.npy'
    val_path = file + f'/val.npy'
    test_path = file + f'/test.npy'
    return train_path, val_path, test_path

def save2file_meta(params, file_name, head):
    def timestamp2str(stamp):
        utc_t = int(stamp)
        utc_h = utc_t // 3600
        utc_m = (utc_t // 60) - utc_h * 60
        utc_s = utc_t % 60
        hour = (utc_h + 8) % 24
        t = f'{hour}:{utc_m}:{utc_s}'
        return t

    import csv, time, os
    dir_check(file_name)
    if not os.path.exists(file_name):
        f = open(file_name, "w", newline='\n')
        csv_file = csv.writer(f)
        csv_file.writerow(head)
        f.close()

    with open(file_name, "a", newline='\n') as file:
        csv_file = csv.writer(file)

        params['log_time'] = timestamp2str(time.time())
        data = [params[k] for k in head]
        csv_file.writerow(data)


# ----- Training Utils----------
import argparse
import random, torch
from torch.optim import Adam
from pprint import pprint
from torch.utils.data import DataLoader

def get_common_params():
    # Training settings
    parser = argparse.ArgumentParser(description='Entry Point of the code')
    parser.add_argument('--is_test', type=bool, default=False, help='test the code')

    # dataset
    parser.add_argument('--min_task_num', type=int, default=0, help='minimal number of task')
    parser.add_argument('--max_task_num', type=int, default=30, help='maxmal number of task')
    parser.add_argument('--eval_min', type=int, default=0, help='minimal number of task')
    parser.add_argument('--eval_max', type=int, default=25, help='maxmal number of task')
    parser.add_argument('--dataset', default='logistics_0831', type=str, help='food_cou or logistics')  # logistics_0831, logistics_decode_mask
    parser.add_argument('--pad_value', type=int, default=29, help='logistics: max_num - 1, pd: max_num + 1')

    ## common settings for deep models
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 256)')
    parser.add_argument('--num_epoch', type=int, default=60, help='number of epochs to train (default: 1000)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 1e-4)')
    parser.add_argument('--seed', type=int, default=2021, metavar='S', help='random seed (default: 6)')
    parser.add_argument('--wd', type=float, default=1e-5, help='weight decay (default: 1e-5)')
    parser.add_argument('--early_stop', type=int, default=10, help='early stop at')
    parser.add_argument('--is_eval', type=str, default=False, help='True means load existing model')
    parser.add_argument('--sort_x_size', type=int, default=8)



    return parser

def to_device(batch, device):
    batch = [x.to(device) for x in batch]
    return batch


def run(params, DATASET, process_batch, test_model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params['pad_value'] = params['max_task_num'] - 1
    params['train_path'], params['val_path'], params['test_path'] = get_dataset_path(params)
    pprint(params)

    train_dataset = DATASET(mode='train', params=params)
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, collate_fn=None, drop_last=False)

    val_dataset = DATASET(mode='val', params=params)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False,collate_fn=None, drop_last=False)

    test_dataset = DATASET(mode='test', params=params)
    test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False, collate_fn=None, drop_last=False)


    PolicyNetwork = Agent.PolicyNetwork(params)

    PolicyNetwork.to(device)

    optimizer = Adam(PolicyNetwork.parameters(), lr=params['lr'], weight_decay=params['wd'])

    if params['target'] == 'krc':
        early_stop = EarlyStop(mode='maximize', patience=params['early_stop'])
    elif params['target'] == 'picp':
        early_stop = EarlyStop(mode='maximize', patience=params['early_stop'])
    elif params['target'] == 'mis':
        early_stop = EarlyStop(mode='minimize', patience=params['early_stop'])
    else:
        early_stop = EarlyStop(mode='minimize', patience=params['early_stop'])

    local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    save_model_path = ws + f'/data/dataset/{params["dataset"]}/params/{params["model"]}_{local_time}.params'
    save_model_dict = ws + f'/data/dataset/{params["dataset"]}/params/'
    dir_check(save_model_dict)
    total_reward = []
    for epoch in range(params['num_epoch']):
        ave_loss = None
        if early_stop.stop_flag: break
        postfix = {"epoch": epoch, "loss": 0.0, "current_loss": 0.0}
        epoch_reward = []
        with tqdm(train_loader, total=len(train_loader), postfix=postfix) as t:
            for i, batch in enumerate(t):
                PolicyNetwork.train()
                loss, batch_reward = process_batch(batch, PolicyNetwork, device, params)

                epoch_reward.append(batch_reward)

                if ave_loss is None:
                    ave_loss = loss.item()
                else:
                    ave_loss = ave_loss * i / (i + 1) + loss.item() / (i + 1)

                postfix["loss"] = ave_loss
                postfix["current_loss"] = loss.item()

                t.set_postfix(**postfix)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_reward.append(epoch_reward)
        val_result = test_model(PolicyNetwork, val_loader, device, params['pad_value'], params,  save2file_, 'val')
        print('\nval result:', val_result.route_eta_to_str(), '| Best epoch:', early_stop.best_epoch)
        is_best_change = early_stop.append(val_result.route_eta_to_dict()['krc'])

        if is_best_change:
            print('best change results:', val_result.route_eta_to_dict(), early_stop.best_metric())
            torch.save(PolicyNetwork.state_dict(), save_model_path)
            print('best model saved')
            print('model path:', save_model_path)
    try:
        print('loaded model path:', save_model_path)
        PolicyNetwork.load_state_dict(torch.load(save_model_path))
        print('best model loaded !!!')
    except:
        print('load best model failed')
    test_result = test_model(PolicyNetwork, test_loader, device, params['pad_value'], params, save2file_, 'test')
    np.save(save_model_dict + f'/reward_{local_time}.npy',np.array(total_reward))
    print('\n-------------------------------------------------------------')
    print('Best epoch: ', early_stop.best_epoch)
    print(f'{params["model"]} Evaluation in test:', test_result.route_eta_to_str())
