import os
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from tqdm import tqdm
from utils.utils import ws, dir_check
from data.preprocess import pre_process, split_trajectory
import argparse
import copy
def idx(df, col_name):
    _idx_ = list(df.columns).index(col_name)
    return _idx_

np.random.seed(1)

from collections import defaultdict
class PickupDataset(object):
    def __init__(self, params):
        self.params = params
        self.N = params['len_range'][1]
        self.N_min = params['len_range'][0]
        self.mode = params['mode']
        self.data = defaultdict(list)
        self.dic_dis_cal = {}

    def dis_cal(self, lat1, lng1, lat2, lng2):
        # distance cache
        key1 = (lat1, lng1, lat2, lng2)
        key2 = (lat2, lng2, lat1, lng1)
        if key1 not in self.dic_dis_cal.keys():
            distance =  int(geodesic((lat1, lng1), (lat2, lng2)).meters)
            self.dic_dis_cal[key1] = distance
        if key2 not in self.dic_dis_cal.keys():
            distance = int(geodesic((lat1, lng1), (lat2, lng2)).meters)
            self.dic_dis_cal[key2] = distance
        return self.dic_dis_cal[key1]

    def get_features(self, cou, mode):
        _len_ = len(cou)
        if _len_ < self.N_min: return 0
        c_v = cou.values
        id_first = c_v[0][self.idx_order_id]  # the first order in the trajectory
        shuffled_list = c_v[1:, self.idx_order_id].tolist()
        np.random.shuffle(shuffled_list)
        shuffled_list = [id_first] + shuffled_list  # set the first order as start node
        #只用拿这段轨迹做一个样本
        V = np.zeros([ self.N, self.params['todo_node_fea_dim']])  # node features
        V_len = np.zeros([0])  # number of visible nodes, including finished and unfinished tasks
        V_pt = np.zeros([self.N])  # promoise time of nodes
        V_ft = np.zeros([self.N])  # finish time of nodes
        V_reach_mask = np.full([ self.N], True)  # mask for reachability of nodes
        V_dispatch_mask = np.zeros([self.N])  # mask for dispathed nodes

        E_mask = np.zeros([self.N, self.N])  # mask for edge connection
        E_abs_dis = np.zeros([self.N, self.N])  # edge feature, absolute distance
        E_rel_dis = np.zeros([self.N, self.N])  # edge feature, relative distance
        E_dt_dif = np.zeros([self.N, self.N])  # edge feature, difference of promised pick-up time
        E_pt_dif = np.zeros([self.N, self.N])  # edge feature, difference of dispatch time
        A = np.zeros([self.N, self.N])  # k-nearest st
        V_non_neighbor = np.full([self.N, self.N], self.N - 1)
        first_node = np.zeros( [ self.params['first_node_fea_dim']])  # features at start node: (dt, lng, lat, pt_last, ft)

        route_label = np.full([self.N], self.N - 1)  # label of route prediction for each step
        time_label = np.zeros([self.N])  # arrival time of nodes
        route_label_len = np.array([0])  # number of valid label nodes for each step
        route_label_all = np.full([self.N], self.N - 1)
        eta_label_len =  np.array([0])
        t_interval = np.zeros([ self.N])  # label: time difference

        start_fea = np.zeros([self.params['start_node_fea_dim']])  # features at start node: (dt, lng, lat, pt_last, ft)
        start_idx = np.array([0])
        past_x = np.zeros([self.params['done_node_num'], self.params['done_node_fea_dim']])  # The features of the previous [done_node_num]-packages of each step
        cou_fea = list(np.array(self.df_cou.loc[self.df_cou['id'] == c_v[0][self.idx_courier_id]]).reshape(-1)[[0, 2, 3, 4, 5, 6, 7, 8, 9]])

        start = 0
        unpick_set = str(c_v[start][self.idx_todo_task])
        if (c_v[start][self.idx_todo_task_num] <= 0) or (unpick_set == 'nan'):
            return 0
        todo_lst = unpick_set.split('.')
        remove_lst = []

        # filter packages
        for task in todo_lst:
            task_id = int(task)
            c_idx = task_id - id_first
            if c_idx <= start:
                remove_lst.append(task)
                print('Error of start index')
                return 0
            # max number of tasks considered
            if c_idx >= self.N - 2:
                remove_lst.append(task)
                if mode == 'test':
                    return 0
                return 0
            # filter packages according to time label
            t_label = c_v[c_idx][self.idx_finish_time] - c_v[start][self.idx_finish_time]

            if t_label <= self.params['label_range'][0] or t_label > self.params['label_range'][1]:
                remove_lst.append(task)
                return 0

        for x in remove_lst:
            todo_lst.remove(x)

        if len(todo_lst) <= self.N_min:
            return 0
        #############################   #############################   #############################
        # After the filtering operation, start to extract features

        # The smaller the index, the earlier it is picked, and the sorted list is obtained by sorting according to the actual picked time.
        todo_rank = list(map(int, todo_lst))
        todo_rank.sort()

        step_valid_order = list(map(int, todo_lst))  # Build features and masks based on valid orders for each step
        for task in todo_rank:
            idx = shuffled_list.index(task)  # return the index of the task in the shuffled list
            V_reach_mask[idx] = False
            V_dispatch_mask[idx] = 1

        V_dispatch_mask[ shuffled_list.index(c_v[:, self.idx_order_id][start])] = 1  # The start node is considered as accepted

        current_time = c_v[start][self.idx_finish_time]  # the time of the current step
        # find the most closed accept_time to current_time
        close_time_dif = np.inf
        close_accept_time = np.inf
        for i in range(len(c_v)):
            accept_t = c_v[i][self.idx_accept_time]
            if (accept_t - current_time > 0) and (accept_t - current_time < close_time_dif): # 在当前出发点时间之后的最早接单时间
                close_time_dif = accept_t - current_time
                close_accept_time = accept_t

        # For label construction, remove package which is influenced by new comming packages, i.e., got_time < close_accept_time
        todo_rank_ = copy.deepcopy(todo_rank)  # 全部todo, 不根据最新接单过滤
        if close_accept_time != np.inf:
            todo_rank = list(filter(lambda k: c_v[k - id_first][self.idx_finish_time] <= close_accept_time, todo_rank)) # 根据新接订单时间过滤

        for task in todo_rank: # route label
            route_label[todo_rank.index(task)] = shuffled_list.index(task)

        for task in todo_rank_:
            task_idx = task - id_first
            t_interval[todo_rank_.index(task)] = c_v[task_idx][self.idx_finish_time] - c_v[task_idx - 1][self.idx_finish_time]
            time_label[todo_rank_.index(task)] = c_v[task_idx][self.idx_finish_time] - c_v[start][self.idx_finish_time]
            route_label_all[todo_rank_.index(task)] = shuffled_list.index(task)

        route_label_true = np.sum((route_label != self.N - 1) + 0)
        route_label_len[0] = len(todo_rank)
        if route_label_true != route_label_len[0]:
            print('label error')
        eta_label_len[0] = len(todo_rank_)

            # iterate all accepted orders, and connect them in the graph
        for i in range(self.N):
            for j in range(self.N):
                if (V_dispatch_mask[i] == 1) and (V_dispatch_mask[j] == 1):  # V_dispatch_mask (T, N)
                    E_mask[i, j] = 1
                    E_mask[j, i] = 1

                # construct features
        for task in todo_lst:
            task_id = int(task)
            c_idx = task_id - id_first
            # The distance between each package and the start point
            dis_abs = self.dis_cal(c_v[start][self.idx_latitude], c_v[start][self.idx_longitude],
                              c_v[c_idx][self.idx_latitude], c_v[c_idx][self.idx_longitude])
            idx = shuffled_list.index(task_id)

            # contruct edge feature
            E_abs_dis[0, idx] = dis_abs
            E_abs_dis[idx, 0] = dis_abs

            if c_v[c_idx][self.idx_courier_distance] == 0:
                dis_temp = 0.0
            else:
                dis_temp = dis_abs / c_v[c_idx][self.idx_courier_distance] * 100

            # if start == 0:
            E_rel_dis[0, idx] = dis_temp
            E_rel_dis[idx, 0] = dis_temp

            # contruct edge feature
            E_dt_dif[0, idx] = c_v[0][self.idx_accept_time] - c_v[c_idx][self.idx_accept_time]
            E_dt_dif[idx, 0] = c_v[c_idx][self.idx_accept_time] - c_v[0][self.idx_accept_time]
            E_pt_dif[0, idx] = c_v[0][self.idx_promised_time] - c_v[c_idx][self.idx_promised_time]
            E_pt_dif[idx, 0] = c_v[c_idx][self.idx_promised_time] - c_v[0][self.idx_promised_time]

            node_feature = c_v[c_idx][[self.idx_accept_time, self.idx_longitude, self.idx_latitude, self.idx_type_code]].tolist()
            node_feature.append(dis_temp)
            node_feature.append(dis_abs)
            node_feature.append(c_v[c_idx][self.idx_promised_time] - c_v[c_idx][self.idx_accept_time])
            node_feature.append(c_v[c_idx][self.idx_promised_time] - c_v[start][self.idx_finish_time])
            node_feature.extend(c_v[c_idx][[self.geohash_3, self.geohash_6]].tolist())
            V[idx, :] = np.array(node_feature)

            V_ft[ idx] = c_v[c_idx][self.idx_finish_time]
            V_pt[ idx] = c_v[c_idx][self.idx_promised_time]

            # get feature of start node, accept_time, lng, lat, pt, ft
            start_fea[:] = c_v[start][[self.idx_accept_time, self.idx_longitude, self.idx_latitude, self.idx_promised_time, self.idx_finish_time, self.geohash_3, self.geohash_6]]
            first_node[ :] = c_v[start][[self.idx_accept_time, self.idx_longitude, self.idx_latitude, self.idx_type_code, 0, 0, self.idx_promised_time - self.idx_accept_time, self.idx_promised_time - self.idx_finish_time]]
            start_idx[0] = shuffled_list.index(c_v[:, self.idx_order_id][start])

            # get features of done_node_num packages
            s, e = max(0, start-2), start + 1 # window start and end
            for idx, n in enumerate(range(s, e)):
                done_feature = [c_v[n][self.idx_finish_time], c_v[n][self.idx_accept_time],
                                 c_v[n][self.idx_longitude], c_v[n][self.idx_latitude], c_v[n][self.idx_type_code],
                                 c_v[n][self.idx_relative_dis_to_last_package], c_v[n][self.idx_dis_to_last_package],
                                 c_v[n][self.idx_time_to_last_package],
                                 c_v[n][self.idx_promised_time] - c_v[n][self.idx_accept_time],
                                 c_v[n][self.geohash_3],
                                 c_v[n][self.geohash_6]]
                past_x[idx, :] = np.array(done_feature)
        # Traversing all steps finished
        ############  ############  ############  ############  ############  ############  ############  ############

        sample_valid_task = list(set(step_valid_order))
        V_len[:] = len(todo_rank_)
        for i in sample_valid_task:
            for j in sample_valid_task:
                idx_i = shuffled_list.index(i)
                idx_j = shuffled_list.index(j)
                c_idx_i = i - id_first
                c_idx_j = j - id_first
                E_dt_dif[idx_i][idx_j] = c_v[c_idx_i][self.idx_accept_time] - c_v[c_idx_j][self.idx_accept_time]
                E_dt_dif[idx_j][idx_i] = c_v[c_idx_j][self.idx_accept_time] - c_v[c_idx_i][self.idx_accept_time]
                E_pt_dif[idx_i][idx_j] = c_v[c_idx_i][self.idx_promised_time] - c_v[c_idx_j][self.idx_promised_time]
                E_pt_dif[idx_j][idx_i] = c_v[c_idx_j][self.idx_promised_time] - c_v[c_idx_i][self.idx_promised_time]
                dis_abs = self.dis_cal(c_v[c_idx_i][self.idx_latitude], c_v[c_idx_i][self.idx_longitude],
                                  c_v[c_idx_j][self.idx_latitude], c_v[c_idx_j][self.idx_longitude])
                if c_v[c_idx_i][self.idx_courier_distance] == 0:
                    dis_temp = 0.0
                else:
                    dis_temp = dis_abs / c_v[c_idx_i][self.idx_courier_distance] * 100
                E_abs_dis[idx_i][idx_j] = dis_abs
                E_abs_dis[idx_j][idx_i] = dis_abs
                E_rel_dis[idx_i][idx_j] = dis_temp
                E_rel_dis[idx_j][idx_i] = dis_temp

        A_reach = ~V_reach_mask + 0
        cur_idx = start_idx[0]
        reachable_nodes = np.argwhere(A_reach == 1).reshape(-1)
        reachable_nodes = np.append(reachable_nodes, [cur_idx])
        for i in range(self.N):
            if i in reachable_nodes:
                for j in range(self.N):
                    if j in reachable_nodes:
                        if i != j:
                            A[i][j] = 1
                        else:
                            A[i][j] = -1

        A_ = copy.deepcopy(A)
        for i in range(self.N):
            if len(np.argwhere(A[i] == 1).reshape(-1)) < 5:
                continue
            dis_from_i = E_abs_dis[i] * A[i]
            remove_dis_idx = np.argsort(abs(dis_from_i))[-1]
            time_from_i = E_pt_dif[i] * A[i]
            remove_time_idx = np.argsort(abs(time_from_i))[-1]
            A[i][[remove_dis_idx, remove_time_idx]] = 0

        for i in range(self.N):
            if len(np.argwhere(A_[i] == 1).reshape(-1)) < 5:
                continue
            valid_index = (np.argwhere(A_[i] == 1).reshape(-1)).tolist()  # 可达节点索引
            dis_from_i = E_abs_dis[i] * A_[i]
            sorted_idx = np.argsort(abs(np.array(dis_from_i))).tolist()
            for j in range(len(dis_from_i)):
                if j not in valid_index:
                    dis_from_i[j] = 99999
            if 6 <= len(np.argwhere(A_[i] == 1).reshape(-1)) <= 7:
                non_neighbor_idx = [x for x in sorted_idx if x in valid_index][len(valid_index) - 1:]
            elif 7 < len(np.argwhere(A_[i] == 1).reshape(-1)) <= 9:
                non_neighbor_idx = [x for x in sorted_idx if x in valid_index][len(valid_index) - 3:]
            else:
                non_neighbor_idx = [x for x in sorted_idx if x in valid_index][len(valid_index) - 4:]

            num_non_neighbor_idx = len(non_neighbor_idx)
            V_non_neighbor[i, 0:num_non_neighbor_idx] = non_neighbor_idx

        #concatenate edge feature
        E_fea_lst = [np.expand_dims(x, axis=-1) for x in [E_abs_dis, E_rel_dis, E_pt_dif, E_dt_dif]]
        E_static_fea = np.concatenate(E_fea_lst, axis=2)
        return {'V': V, 'V_reach_mask': V_reach_mask, 'V_dispatch_mask': V_dispatch_mask, 'V_len': V_len, 'V_pt': V_pt,
                'V_ft': V_ft,
                'E_mask': E_mask, 'E_static_fea': E_static_fea, 'A': A,
                'start_fea': start_fea, 'start_idx': start_idx,
                'past_x': past_x, 'cou_fea': cou_fea,
                'route_label': route_label, 'time_label': time_label, 'label_len': route_label_len,
                't_interval': t_interval, 'eta_label_len': eta_label_len,
                'route_label_all': route_label_all,
                'V_non_neighbor': V_non_neighbor, 'first_node': first_node}

    def results_merge(self, results):

        all_result = []
        for r in results:
            all_result += r

        feature_lst = ['V', 'V_len', 'V_pt', 'V_ft',  'V_reach_mask', 'V_dispatch_mask', # node/task related information
                      'E_mask', 'A', 'E_static_fea', # edge related information
                      'start_fea', 'start_idx', # start node information,
                      'past_x', 'cou_fea', # past task, and courier's feature
                      'route_label', 'time_label', 'label_len', 't_interval',  'eta_label_len', 'route_label_all', 'V_non_neighbor', 'first_node' ]  #label information,

        for r in all_result:
            for f in feature_lst:
                self.data[f].append(r[f])

        for f in feature_lst:
            self.data[f] = np.stack(self.data[f], axis=0)


        return self.data

    def multi_thread_work(self, parameter_queue, function_name, thread_number=5):
        from multiprocessing import Pool
        """
        For parallelization
        """
        pool = Pool(thread_number)
        result = pool.map(function_name, parameter_queue)
        pool.close()
        pool.join()
        return result

    def get_feature_kernel(self, args={}):
        c_lst = args['c_lst']
        pbar = tqdm(total=len(c_lst))
        result_lst = []

        for cou in c_lst:
            pbar.update(1)
            for i in range(len(cou)):
                if (i + self.N <= len(cou)) and (i > 0):
                    result = self.get_features(cou[i:i + self.N - 1], self.mode)
                elif  (i + self.N > len(cou)) and (i == 0):
                    result = self.get_features(cou, self.mode)
                elif (i + self.N > len(cou)) and (i > 0):
                    result = self.get_features(cou[i: len(cou)], self.mode)
                elif (i + self.N <= len(cou)) and (i == 0):
                    result = self.get_features(cou[i: i + self.N - 1], self.mode)
                else:
                    result = 0

                if result != 0:
                    result_lst.append(result)

        return result_lst

    def get_data(self):
        if os.path.exists(self.params['fin_temp']):
            self.df = pd.read_csv(self.params['fin_temp'] + "/package_feature.csv", sep=',', encoding='utf-8')
            self.df_cou = pd.read_csv(self.params['fin_temp'] + "/courier_feature.csv", sep=',', encoding='utf-8')
        else:
            self.df, self.df_cou = pre_process(fin=self.params['fin_original'], fout=self.params['fin_temp'], is_test=self.params['is_test'])
        # feature index in the dataframe
        self.idx_order_id = idx(self.df, 'index')
        self.idx_courier_id = idx(self.df, 'courier_id')
        self.idx_todo_task_num = idx(self.df, 'todo_task_num')
        self.idx_todo_task = idx(self.df, 'todo_task')
        self.idx_finish_time = idx(self.df, 'finish_time_minute')
        self.idx_accept_time = idx(self.df, 'accept_time_minute')
        self.idx_promised_time = idx(self.df, 'expect_finish_time_minute')
        self.idx_longitude = idx(self.df, 'lng')
        self.idx_latitude = idx(self.df, 'lat')
        self.idx_type_code = idx(self.df, 'aoi_type')
        self.idx_courier_distance = idx(self.df, 'dis_avg_day')
        self.idx_relative_dis_to_last_package = idx(self.df, 'relative_dis_to_last_package')
        self.idx_dis_to_last_package = idx(self.df, 'dis_to_last_package')
        self.idx_time_to_last_package = idx(self.df, 'time_to_last_package')
        self.idx_abs_got_time = idx(self.df, 'pickup_time')
        self.geohash_6 = idx(self.df, 'geohash_6')
        self.geohash_3 = idx(self.df, 'geohash_3')
        all_dates = self.df['ds'].unique()
        self.idx_original_order_id = idx(self.df, 'order_id')
        train_dates = all_dates[: int(len(all_dates) * self.params['train_ratio'])].tolist()
        val_dates = all_dates[int(len(all_dates) * self.params['train_ratio']) + 1: int(len(all_dates) * (self.params['train_ratio'] + self.params['val_ratio']))].tolist()
        test_dates = all_dates[int(len(all_dates) * 0.8) + 1:].tolist()
        if self.params['is_test'] == False:

            if self.mode == 'train':
                self.df = self.df[self.df['ds'].isin(train_dates)]
            elif self.mode == 'val':
                self.df = self.df[self.df['ds'].isin(val_dates)]
            elif self.mode == 'test':
                self.df = self.df[self.df['ds'].isin(test_dates)]
            else:
                raise RuntimeError('please specify a mode')

        courier_l = split_trajectory(self.df)
        thread_num = self.params['num_thread']

        if self.params['is_test']:
            courier_l = courier_l[:10]
            thread_num = 2

        # courier_l = courier_l[:300]

        n = len(courier_l)
        task_num = n // thread_num
        args_lst = [{'c_lst': courier_l[i: min(i + task_num, n)]} for i in range(0, n, task_num)]
        results_list = self.multi_thread_work(args_lst, self.get_feature_kernel, self.params['num_thread'])
        return self.results_merge(results_list)

def get_params():

    parser = argparse.ArgumentParser()
    # dataset parameters
    parser.add_argument('--fin_temp', type=str, default=ws + '/data/tmp/pickup_yt/')
    parser.add_argument('--fin_original',  type=str, default=ws + '/data/raw//pickup_yt.csv')
    parser.add_argument('--data_name', type=str, default='pickup_yt')
    parser.add_argument('--num_thread', type=int, default=20)
    parser.add_argument('--random_seed', type=int, default=1)

    parser.add_argument('--is_test', type=str, default=False)
    parser.add_argument('--label_range', type=tuple, default=(0, 180))
    parser.add_argument('--len_range', type=tuple, default=(2, 30))
    parser.add_argument('--todo_node_fea_dim', type=int, default=10, help='feature number for each unfinished task node')
    parser.add_argument('--start_node_fea_dim', type=int, default=7, help='feature number for start node')
    parser.add_argument('--done_node_fea_dim', type=int, default=11, help='feature number for each finished node')
    parser.add_argument('--done_node_num', type=int, default=3, help='the number of last finished nodes')
    parser.add_argument('--first_node_fea_dim', type=int, default=8, help='feature number for first node')

    # data split, according to date
    parser.add_argument('--train_ratio', type=float, default=0.6)
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    args, _ = parser.parse_known_args()
    return args


def main():
    params = vars(get_params())
    if params['is_test']: params['data_name'] += '_test'
    data_name = params['data_name']

    for mode in ['train', 'val', 'test']:
        if mode == 'train':
            fout = ws + f'/data/dataset/{data_name}/train.npy'
            dir_check(fout)
            params['mode'] = mode
            DataProcess = PickupDataset(params)
            np.save(fout, DataProcess.get_data())
            print('train file saved at: ', fout)
        elif mode == 'val':
            fout = ws + f'/data/dataset/{data_name}/val.npy'
            dir_check(fout)
            params['mode'] = mode
            DataProcess = PickupDataset(params)
            np.save(fout, DataProcess.get_data())
            print('val file saved at: ', fout)
        else:
            fout = ws + f'/data/dataset/{data_name}/test.npy'
            dir_check(fout)
            params['mode'] = mode
            DataProcess = PickupDataset(params)
            np.save(fout, DataProcess.get_data())
            print('test file saved at: ', fout)

    print('Dataset constructed...')


if __name__ == "__main__":

    params = vars(get_params())
    main(params)




