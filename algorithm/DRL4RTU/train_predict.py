import numpy as np
import torch.nn.functional as F
from tqdm import  tqdm
import torch
from utils.utils import to_device, dict_merge
from algorithm.DRL4RTU.Dataset import DRL4RTU_dataset


def get_nonzeros_eta(pred_steps, label_steps, label_len, sigma_pred, eta_pred, eta_label, pad_value):
    pred = []
    label = []
    label_len_list = []
    sigma_pred_list = []
    eta_pred_list = []
    eta_label_list = []

    for i in range(pred_steps.size()[0]):
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

def test_model(model, test_dataloader, device, pad_value, params, save2file, mode):
    from utils.eval import Metric
    model.eval()

    evaluator_1 = Metric([1, 5])
    evaluator_2 = Metric([1, 11])
    evaluator_3 = Metric([1, 15])
    evaluator_4 = Metric([1, 25])

    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            batch = to_device(batch, device)
            V, V_reach_mask, label, label_len, V_at, start_fea, start_idx, eta_label_len, route_label_all, pred_len, first_node = batch
            V, V_reach_mask, label, label_len, V_at, start_fea, eta_label_len, start_idx, pred_len, route_label_all, first_node = filter_input(
                V, V_reach_mask, label, label_len, V_at, start_fea, eta_label_len, start_idx, route_label_all,
                first_node)

            outputs, result_route, eta_predict, sigma_pred = model(V, V_reach_mask, start_fea, first_node, decode_type='mle')

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
def eta_maemis_loss_calc(time_label, eta_label_len, eta, sigma, params):

    pred_tensor = torch.empty(0).to(time_label.device)
    label_tensor = torch.empty(0).to(time_label.device)
    pred_upper_tensor = torch.empty(0).to(time_label.device)
    pred_lower_tensor = torch.empty(0).to(time_label.device)
    for i in range(len(eta_label_len)):
        lab_len = int(eta_label_len[i].item())
        lab = time_label[i][:lab_len]
        pre = eta[i][:lab_len]
        sigm = sigma[i][:lab_len]
        pred_upper = pre + sigm
        pred_lower = pre - sigm
        pred_tensor = torch.cat([pred_tensor, pre])
        label_tensor = torch.cat([label_tensor, lab])
        pred_upper_tensor = torch.cat([pred_upper_tensor, pred_upper])
        pred_lower_tensor = torch.cat([pred_lower_tensor, pred_lower])

    loss0 = criterion_tp(pred_tensor, label_tensor) * 10
    loss1 = torch.mean(torch.max(pred_upper_tensor - pred_lower_tensor, torch.tensor([0.]).to(time_label.device)))
    loss2 = torch.mean(torch.max(pred_lower_tensor - label_tensor.float(), torch.tensor([0.]).to(time_label.device))) * params['cred']
    loss3 = torch.mean(torch.max(label_tensor.float() - pred_upper_tensor, torch.tensor([0.]).to(time_label.device))) * params['cred']
    return loss0 + loss1 + loss2 + loss3


def calc_route_time_reward(route_prediction, route_label, label_len, rt_log_probs, pred_lens, eta_prediction, sigma_prediction, eta_label, eta_label_len, route_label_all, params):
    from utils.eval import location_deviation, hit_rate, route_acc
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

    rt_log_probs_saved = []
    pred_len_saved = []
    rt_reward = []

    route_prediction, route_label, label_len, eta_prediction, sigma_prediction, eta_label, eta_label_len, route_label_all =\
        [tensor2lst(x) for x in [route_prediction, route_label, label_len, eta_prediction, sigma_prediction, eta_label, eta_label_len, route_label_all]]

    for pre, lab, lab_len, rt_log_prob, pred_len, eta_pred, sigma_pred, eta_lab, eta_lab_len, route_lab_all in zip(route_prediction, route_label, label_len, rt_log_probs, pred_lens, eta_prediction, sigma_prediction, eta_label, eta_label_len, route_label_all):
        if lab_len == 0:
            continue
        else:
            uq_step_reward = uq_reward_calc(eta_pred, sigma_pred, eta_lab, lab, lab_len, params)
            if params['reward_type'] == 'hr1+lsd+picp':
                route_step_reward = params['R'] - location_deviation(pre, lab, lab_len, 'square')  + params['beta'] * hit_rate(pre, lab, lab_len, top_n=1)
                sequence_reward = route_step_reward * (uq_step_reward + params['lambda'])
                rt_reward.append(sequence_reward)
                seq_lab_len = lab_len * 3
                seq_log_prob = rt_log_prob
            elif params['reward_type'] == 'acc3+picp':
                route_step_reward = route_acc(pre, lab[:lab_len], 3) * 10
                sequence_reward = route_step_reward * uq_step_reward + route_step_reward + uq_step_reward
                rt_reward.append(sequence_reward)
                seq_lab_len = lab_len * 3
                seq_log_prob = rt_log_prob
            elif params['reward_type'] == 'lsd + picp':
                acc_reward = route_acc(pre, lab[:lab_len], 3) * 50
                lsd_reward = 100 - location_deviation(pre, lab, lab_len, 'square')
                if acc_reward + lsd_reward > 0:
                    sequence_reward = (acc_reward + lsd_reward) * uq_step_reward
                else:
                    sequence_reward = 0
                rt_reward.append(sequence_reward)
                seq_lab_len = lab_len * 3
                seq_log_prob = rt_log_prob

            rt_log_probs_saved.append(seq_log_prob)
            pred_len_saved.append(torch.tensor(seq_lab_len).to(pred_len.device))

    return np.array(rt_reward), torch.stack(rt_log_probs_saved, dim=0), torch.stack(pred_len_saved, dim=0)


def get_log_prob_mask(pred_len, params):
    if params['reward_type'] == 'hr1+lsd+picp':
        log_prob_mask = torch.zeros([pred_len.shape[0], params['max_task_num'] * 3]).to(pred_len.device)
    elif params['reward_type'] == 'acc3+picp':
        log_prob_mask = torch.zeros([pred_len.shape[0], params['max_task_num'] * 3]).to(pred_len.device)
    elif params['reward_type'] == 'joint_reward':
        log_prob_mask = torch.zeros([pred_len.shape[0], params['max_task_num'] * 3]).to(pred_len.device)

    for i in range(len(pred_len)):
        valid_len = pred_len[i].long().item()
        log_prob_mask[i][:valid_len] = 1
    return log_prob_mask

def filter_input(V, V_reach_mask, label, label_len, V_at, start_fea, eta_label_len,  start_idx, route_label_all, first_node):
    B, T, N = V_reach_mask.shape[0], V_reach_mask.shape[1], V_reach_mask.shape[2]
    valid_len = torch.sum((~(V.reshape(-1, N, V.shape[-1])[:, :, 0] == 0) + 0).reshape(B * T, N), dim=1) #根据V选有效节点，也是eta有效的，label len可能为0
    valid_index = ((valid_len != 0) + 0).nonzero().squeeze(1)
    V = V.reshape(B * T, N, -1)[valid_index]
    label = label.reshape(B * T, N)[valid_index]
    label_len = label_len.reshape(B * T)[valid_index]
    V_at = V_at.reshape(B * T, N)[valid_index]
    start_fea = start_fea.reshape(B * T, -1)[valid_index]
    V_reach_mask = V_reach_mask.reshape(B * T, N)[valid_index]
    eta_label_len = eta_label_len.reshape(B * T, -1)[valid_index]
    start_idx = start_idx.reshape(B*T, -1)[valid_index]
    valid_len = valid_len.reshape(-1)[valid_index]
    route_label_all = route_label_all.reshape(B * T, N)[valid_index]
    first_node = first_node.reshape(B * T, -1)[valid_index]

    return V, V_reach_mask, label, label_len, V_at, start_fea, eta_label_len, start_idx, valid_len, route_label_all, first_node

def process_batch(batch, model, device, params):
    batch = to_device(batch, device)
    V, V_reach_mask, label, label_len, V_at, start_fea, start_idx, eta_label_len, route_label_all, pred_len, first_node = batch
    V, V_reach_mask, label, label_len, V_at, start_fea, eta_label_len, start_idx, pred_len, route_label_all, first_node = filter_input(
        V, V_reach_mask, label, label_len, V_at, start_fea, eta_label_len, start_idx, route_label_all,
        first_node)

    #mle
    pred_scores, pred_pointers, eta_select, sigma_select = model(V, V_reach_mask, start_fea,  first_node, decode_type = 'mle')
    unrolled = pred_scores.view(-1, pred_scores.size(-1))
    mle_loss = F.cross_entropy(unrolled, label.view(-1), ignore_index=params['pad_value'])
    eta_loss = eta_maemis_loss_calc(V_at, label_len, eta_select, sigma_select, params)

    #rl
    route_sample, eta_sample, sigma_samples, order_log_probs, eta_log_probs, sigma_log_probs, rt_log_probs = model(V, V_reach_mask, start_fea, first_node, decode_type='rl')
    with torch.autograd.no_grad():
        _, greedy_pred_pointers, eta_greedy, sigma_greedy = model(V, V_reach_mask, start_fea, first_node, decode_type='mle')

    baseline_rt, rt_log_probs, rt_pred_len_filtered = calc_route_time_reward(greedy_pred_pointers, label, label_len.long(), rt_log_probs, pred_len, eta_greedy, sigma_greedy, V_at, eta_label_len.long(), route_label_all, params)
    sample_rt, _, _ = calc_route_time_reward(route_sample, label, label_len.long(), torch.zeros(greedy_pred_pointers.shape[0], params['max_task_num'] * 3), eta_label_len.long(), eta_sample, sigma_samples, V_at, eta_label_len.long(), route_label_all, params)

    log_prob_mask = get_log_prob_mask(rt_pred_len_filtered, params)
    rt_log_probs = rt_log_probs * log_prob_mask
    rt_log_probs = torch.sum(rt_log_probs, dim=1) / rt_pred_len_filtered

    #SCST loss
    rt_policy_loss = -torch.mean(torch.tensor(sample_rt - baseline_rt).to(rt_log_probs.device) * rt_log_probs)

    return eta_loss + mle_loss + rt_policy_loss * params['rl_ratio'], sample_rt

def main(params):

    from utils.utils import run
    run(params, DRL4RTU_dataset, process_batch, test_model)

def get_params():
    from utils.utils import get_common_params
    parser = get_common_params()
    args, _ = parser.parse_known_args()
    return args

