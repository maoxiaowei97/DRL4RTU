"""
1.去掉了时间步
2.根据起终点相似轨迹特征，通过索引特征获取相似轨迹、标准差
3.模型，输出标准差
"""
import numpy as np
import torch.nn.functional as F
from tqdm import  tqdm
import torch
from my_utils.utils_DRL4Route import to_device, dict_merge
from algorithm.DRL4RTP_1023.Dataset import DRL4RouteDataset


def get_nonzeros_eta(pred_steps, label_steps, label_len, sigma_pred, eta_pred, eta_label, pad_value):
    pred = []
    label = []
    label_len_list = []
    sigma_pred_list = []
    eta_pred_list = []
    eta_label_list = []

    for i in range(pred_steps.size()[0]):
        # label 不为0时才会考虑该测试该step
        if label_steps[i].min().item() != pad_value:
            label.append(label_steps[i].cpu().numpy().tolist())
            pred.append(pred_steps[i].cpu().numpy().tolist())
            label_len_list.append(label_len[i].cpu().numpy().tolist())
            sigma_pred_list.append(sigma_pred[i].cpu().numpy().tolist())
            eta_pred_list.append(eta_pred[i].cpu().numpy().tolist())
            eta_label_list.append(eta_label[i].cpu().numpy().tolist())
    return torch.LongTensor(pred), torch.LongTensor(label), \
           torch.LongTensor(label_len_list), torch.LongTensor(sigma_pred_list),\
           torch.LongTensor(eta_pred_list), torch.LongTensor(eta_label_list)

def get_eta_result(pred, label, label_len, label_route,  pred_pointers, eta_label_len, route_label_all, sigma):
    N = label_route.shape[1]
    B = label_route.shape[0]
    eta_pred_result = torch.zeros(B, N).to(label.device)
    eta_sigma_result = torch.zeros(B, N).to(label.device)
    eta_label_result = torch.zeros(B, N).to(label.device)
    label_len = label_len.reshape(B)
    eta_label_len = eta_label_len.reshape(B)
    label = label.reshape(B, N)

    label_len_list = []
    eta_label_len_list = []
    eta_pred_list = []
    eta_sigma_list = []
    eta_label_list = []
    route_pred_list = []
    route_label_list = []

    label_route = label_route.reshape(B, N)
    for i in range(B):#将eta预测值按路线预测的顺序取出
        if label_len[i].long().item() != 0:
            eta_pred_result[i][:label_len[i].long().item()] = pred[i][label_route[i][:label_len[i].long().item()].long()]  # 根据label_route中各个节点的真实顺序按索引取出预测值，再与按真实顺序排的eta对比
            eta_label_result[i][:label_len[i].long().item()] = label[i][:label_len[i].long().item()]
            eta_sigma_result[i][:label_len[i].long().item()] = sigma[i][label_route[i][:label_len[i].long().item()].long()]

    for i in range(B): #过滤了label_len为0的样本, 9.22. 23:30改为过滤eta_label_len为零的样本, 9.23 10:13改为过滤label_len为0的样本
        if label_len[i].long().item() != 0:
            eta_label_list.append(eta_label_result[i].detach().cpu().numpy().tolist())
            eta_pred_list.append(eta_pred_result[i].detach().cpu().numpy().tolist())
            label_len_list.append(label_len[i].detach().cpu().numpy().tolist())
            route_pred_list.append(pred_pointers[i].detach().cpu().numpy().tolist())
            route_label_list.append(label_route[i].detach().cpu().numpy().tolist())
            eta_label_len_list.append(eta_label_len[i].detach().cpu().numpy().tolist())
            eta_sigma_list.append(eta_sigma_result[i].detach().cpu().numpy().tolist())

    return  torch.LongTensor(label_len_list), torch.tensor(eta_pred_list), torch.tensor(eta_label_list),\
            torch.LongTensor(route_pred_list), torch.LongTensor(route_label_list), torch.LongTensor(eta_label_len_list), torch.tensor(eta_sigma_list)

def test_model(model, test_dataloader, device, pad_value, params, save2file, mode):
    from my_utils.eval_DRL4RTP_selfcritic import Metric
    model.eval()

    evaluator_1 = Metric([1, 5])
    evaluator_2 = Metric([1, 11])
    evaluator_3 = Metric([1, 15])
    evaluator_4 = Metric([1, 25])

    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            batch = to_device(batch, device)
            V, V_reach_mask, label, label_len, V_at, start_fea, V_non_neighbor, start_idx, cou_fea, eta_label_len, first_node, route_label_all, pred_len = batch

            outputs, result_route, eta_resort, sigma_resort, eta_predict, sigma_pred, sigma_label_resort = model(V[:, :, :-2], V_reach_mask, start_fea[:, :-2], cou_fea,
                                                                 pred_len, first_node, start_fea[:, -2], V[:, :, -2], decode_type='mle')


            # pred_eta_save.append(eta_pred.detach().cpu().numpy())
            # label_eta_save.append(eta_label.detach().cpu().numpy())
            # label_len_save.append(label_len.detach().cpu().numpy())
            # eta_label_len_save.append(eta_label_len.detach().cpu().numpy())
            # eta_sigma_save.append(eta_sigma.detach().cpu().numpy())
            route_pred, route_label, labels_len, sigma_pred, eta_preds, eta_label = get_nonzeros_eta(
                result_route.reshape(-1, label.size(-1)), label.reshape(-1, label.size(-1)), label_len.reshape(-1),
                sigma_pred.reshape(-1, label.size(-1)), eta_predict.reshape(-1, label.size(-1)),
                V_at.reshape(-1, label.size(-1)), params['pad_value'])

            evaluator_1.update_route_eta(route_pred, route_label, labels_len, eta_preds, sigma_pred, eta_label,
                                         labels_len)
            evaluator_2.update_route_eta(route_pred, route_label, labels_len, eta_preds, sigma_pred, eta_label,
                                         labels_len)
            evaluator_3.update_route_eta(route_pred, route_label, labels_len, eta_preds, sigma_pred, eta_label,
                                         labels_len)
            evaluator_4.update_route_eta(route_pred, route_label, labels_len, eta_preds, sigma_pred, eta_label,
                                         labels_len)
    if mode == 'val':
        return evaluator_4

    params_1 = dict_merge([evaluator_1.route_eta_to_dict(), params])
    params_1['eval_min'] = 1
    params_1['eval_max'] = 5
    save2file(params_1)

    print(evaluator_2.route_eta_to_str())
    params_2 = dict_merge([evaluator_2.route_eta_to_dict(), params])
    params_2['eval_min'] = 1
    params_2['eval_max'] = 11
    save2file(params_2)

    print(evaluator_3.route_eta_to_str())
    params_3 = dict_merge([evaluator_3.route_eta_to_dict(), params])
    params_3['eval_min'] = 1
    params_3['eval_max'] = 15
    save2file(params_3)

    print(evaluator_4.route_eta_to_str())
    params_4 = dict_merge([evaluator_4.route_eta_to_dict(), params])
    params_4['eval_min'] = 1
    params_4['eval_max'] = 25
    save2file(params_4)


    return evaluator_4
import torch.nn as nn
mse_loss = nn.MSELoss()
criterion_tp = nn.L1Loss()
def eta_maemis_loss_calc(time_label, eta_label_len, eta, sigma, label_route, sigma_label, params): # eta_label_len在此为route_label_len, 考虑新增订单影响, V_at: route label

    pred_tensor = torch.empty(0).to(time_label.device)
    label_tensor = torch.empty(0).to(time_label.device)
    pred_upper_tensor = torch.empty(0).to(time_label.device)
    pred_lower_tensor = torch.empty(0).to(time_label.device)
    for i in range(len(eta_label_len)):
        lab_len = eta_label_len[i][0]
        lab = time_label[i][:lab_len]
        pre = eta[i][:lab_len]
        sigm = sigma[i][:lab_len]
        pred_upper = pre + sigm
        pred_lower = pre - sigm
        pred_tensor = torch.cat([pred_tensor, pre])
        label_tensor = torch.cat([label_tensor, lab])
        pred_upper_tensor = torch.cat([pred_upper_tensor, pred_upper])
        pred_lower_tensor = torch.cat([pred_lower_tensor, pred_lower])

    loss0 = criterion_tp(pred_tensor, label_tensor)
    loss1 = torch.mean(torch.max(pred_upper_tensor - pred_lower_tensor, torch.tensor([0.]).to(time_label.device)))  # upper > lower时惩罚
    loss2 = torch.mean(torch.max(pred_lower_tensor - label_tensor.float(), torch.tensor([0.]).to(time_label.device))) * params['cred']  # lower > label时惩罚
    loss3 = torch.mean(torch.max(label_tensor.float() - pred_upper_tensor, torch.tensor([0.]).to(time_label.device))) * params['cred']  # label > upper时惩罚
    return loss0 + loss1 + loss2 + loss3


def calc_route_time_reward(route_prediction, route_label, label_len, rt_log_probs, pred_lens, eta_prediction, sigma_prediction, eta_label, eta_label_len, route_label_all, params):
    from my_utils.eval_DRL4RTP_selfcritic import location_deviation, hit_rate, route_acc
    def uq_reward_calc(eta_pred, sigma_pred, eta_lab, route_label, label_len, params):
        #PICP
        eta = np.array(eta_pred)[route_label[:label_len]]
        sigma = np.array(sigma_pred)[route_label[:label_len]]
        label = np.array(eta_lab[:label_len])
        upper = eta + sigma
        lower = eta - sigma
        count = np.sum((label > lower) & (label < upper))
        return count / label_len

    def tensor2lst(x):
        try:
            return x.detach().cpu().numpy().tolist()
        except:
            return x

    B, N = route_prediction.shape[0], route_prediction.shape[1]
    rt_log_probs_saved = []
    pred_len_saved = []
    rt_reward = []

    route_prediction, route_label, label_len, eta_prediction, sigma_prediction, eta_label, eta_label_len, route_label_all =\
        [tensor2lst(x) for x in [route_prediction, route_label, label_len, eta_prediction, sigma_prediction, eta_label, eta_label_len, route_label_all]]

    for pre, lab, lab_len, rt_log_prob, pred_len, eta_pred, sigma_pred, eta_lab, eta_lab_len, route_lab_all in zip(route_prediction, route_label, label_len, rt_log_probs, pred_lens, eta_prediction, sigma_prediction, eta_label, eta_label_len, route_label_all):
        if lab_len[0] == 0:
            continue
        else:
            uq_step_reward = uq_reward_calc(eta_pred, sigma_pred, eta_lab, lab, lab_len[0], params) * 10  # 越大越好
            if params['reward_type'] == 'lsd+picp':
                route_step_reward = 100 - location_deviation(pre, lab, lab_len[0], 'square')
                sequence_reward = route_step_reward * uq_step_reward + route_step_reward + uq_step_reward
                rt_reward.append(sequence_reward)
                seq_lab_len = lab_len[0] * 3
                seq_log_prob = rt_log_prob
            elif params['reward_type'] == 'lsd':
                route_step_reward = 100 - location_deviation(pre, lab, lab_len[0], 'square')
                sequence_reward = route_step_reward
                rt_reward.append(sequence_reward)
                seq_lab_len = lab_len[0]
                seq_log_prob = rt_log_prob[torch.arange(0, params['max_task_num'] * 3, 3)]
            elif params['reward_type'] == 'picp':
                sequence_reward = uq_step_reward
                rt_reward.append(sequence_reward)
                seq_lab_len = lab_len[0] * 3
                seq_log_prob = rt_log_prob
            elif params['reward_type'] == 'acc3':
                route_step_reward = route_acc(pre, lab[:lab_len[0]], 3) * 100
                sequence_reward = route_step_reward
                rt_reward.append(sequence_reward)
                seq_lab_len = lab_len[0]
                seq_log_prob = rt_log_prob[torch.arange(0, params['max_task_num'] * 3, 3)]
            elif params['reward_type'] == 'acc3+picp':
                route_step_reward = route_acc(pre, lab[:lab_len[0]], 3) * 10
                sequence_reward = route_step_reward * uq_step_reward + route_step_reward + uq_step_reward
                rt_reward.append(sequence_reward)
                seq_lab_len = lab_len[0] * 3
                seq_log_prob = rt_log_prob
            rt_log_probs_saved.append(seq_log_prob)
            pred_len_saved.append(torch.tensor(seq_lab_len).to(pred_len.device))

    return np.array(rt_reward), torch.stack(rt_log_probs_saved, dim=0), torch.stack(pred_len_saved, dim=0)


def get_log_prob_mask(pred_len, params):
    if params['reward_type'] == 'lsd+picp':
        log_prob_mask = torch.zeros([pred_len.shape[0], params['max_task_num'] * 3]).to(pred_len.device)
    elif params['reward_type'] == 'lsd':
        log_prob_mask = torch.zeros([pred_len.shape[0], params['max_task_num']]).to(pred_len.device)
    elif params['reward_type'] == 'picp':
        log_prob_mask = torch.zeros([pred_len.shape[0], params['max_task_num'] * 3]).to(pred_len.device)
    elif params['reward_type'] == 'acc3':
        log_prob_mask = torch.zeros([pred_len.shape[0], params['max_task_num']]).to(pred_len.device)
    elif params['reward_type'] == 'acc3+picp':
        log_prob_mask = torch.zeros([pred_len.shape[0], params['max_task_num'] * 3]).to(pred_len.device)

    for i in range(len(pred_len)):
        valid_len = pred_len[i].long().item()
        log_prob_mask[i][:valid_len] = 1
    return log_prob_mask


def process_batch(batch, model, device, params):
    batch = to_device(batch, device)
    V, V_reach_mask, label, label_len, V_at, start_fea, V_non_neighbor, start_idx, cou_fea, eta_label_len, first_node, route_label_all, pred_len = batch
    #mle
    pred_scores, pred_pointers, eta_resort, sigma_resort, eta_select, sigma_select, sigma_label_resort = model(V[:, :, :-2], V_reach_mask, start_fea[:, :-2], cou_fea, pred_len, first_node, start_fea[:, -2], V[:, :, -2], decode_type = 'mle')
    unrolled = pred_scores.view(-1, pred_scores.size(-1))
    mle_loss = F.cross_entropy(unrolled, label.view(-1), ignore_index=params['pad_value'])
    eta_loss = eta_maemis_loss_calc(V_at, label_len, eta_select, sigma_select, label, sigma_label_resort, params)
    #rl
    route_sample, eta_sample, sigma_samples, order_log_probs, eta_log_probs, sigma_log_probs, rt_log_probs = model(V[:, :, :-2], V_reach_mask, start_fea[:, :-2], cou_fea, pred_len, first_node, start_fea[:, -2], V[:, :, -2], decode_type='rl')
    with torch.autograd.no_grad():
        _, greedy_pred_pointers, _, _, eta_greedy, sigma_greedy, _ = model(V[:, :, :-2], V_reach_mask, start_fea[:, :-2], cou_fea, pred_len, first_node, start_fea[:, -2], V[:, :, -2], decode_type = 'mle')

    baseline_rt, rt_log_probs, rt_pred_len_filtered = calc_route_time_reward(greedy_pred_pointers, label, label_len.long(), rt_log_probs, pred_len, eta_greedy, sigma_greedy, V_at, eta_label_len.long(), route_label_all, params)
    sample_rt, _, _ = calc_route_time_reward(route_sample, label, label_len.long(), torch.zeros(greedy_pred_pointers.shape[0], params['max_task_num'] * 3), torch.zeros(greedy_pred_pointers.shape[0], params['max_task_num'] * 3), eta_sample, sigma_samples, V_at, eta_label_len.long(), route_label_all, params)

    log_prob_mask = get_log_prob_mask(rt_pred_len_filtered, params)
    rt_log_probs = rt_log_probs * log_prob_mask
    rt_log_probs = torch.sum(rt_log_probs, dim=1) / rt_pred_len_filtered

    if params['route_metric'] == 'lsd':
        rt_policy_loss = -torch.mean(torch.tensor(baseline_rt - sample_rt).to(rt_log_probs.device) * rt_log_probs) # 越小越好
    elif (params['route_metric'] == 'acc3') or (params['route_metric'] == 'acc2') or (params['route_metric'] == 'acc1') or (params['route_metric'] == 'hr3'):
        rt_policy_loss = -torch.mean(torch.tensor(sample_rt - baseline_rt).to(rt_log_probs.device) * rt_log_probs)
    else:
        rt_policy_loss = 0

    return eta_loss + mle_loss + rt_policy_loss * params['rl_ratio'], sample_rt

def main(params):
    from my_utils.utils_drl4rtp_self_critic import RL
    rl = RL()
    rl.run(params, DRL4RouteDataset, process_batch, test_model)

def get_params():
    from my_utils.utils_DRL4Route import get_common_params
    parser = get_common_params()
    args, _ = parser.parse_known_args()
    return args

if __name__ == '__main__':
    import time
    import logging

    logger = logging.getLogger('training')
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    try:
        params = vars(get_params())
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise
