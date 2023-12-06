# -*- coding: utf-8 -*-
from pprint import pprint
import warnings
warnings.filterwarnings("ignore")
def run(params):
    pprint(params)
    import algorithm.DRL4RTU.train_predict as DRL4RTU
    DRL4RTU.main(params)

def get_params():
    from utils.utils import get_common_params
    parser = get_common_params()
    args, _ = parser.parse_known_args()
    return args

if __name__ == "__main__":
    from utils.utils import dict_merge
    params = vars(get_params())
    params['batch_size'] = 256
    params['is_test'] = False
    params['inference'] = False
    params['early_stop'] = 10
    params['max_task_num'] = 30

    params['target'] = 'krc'
    params['gated_fusion'] = True

    args_lst = []
    for model in ['DRL4RTU']:
            for hs in [64]:
                for confidence in [1]:
                    for topk in [1]:
                            for trace_decay in [0.99]:
                                for dataset in ['pickup_yt']:
                                        for rl_r in [0.5, 1]:
                                            for cred in [20]:
                                                for fusion in [True]:
                                                    for reward_type in ['acc3+picp', 'lsd+picp', 'joint_reward']:
                                                        for target in ['krc', 'mis']:
                                                            params_dict = {'model': model, 'hidden_size': hs, 'rl_ratio':rl_r,  'reward_type': reward_type,
                                                                               'trace_decay': trace_decay, 'cred':cred, 'gated_fusion': fusion,  'if_fusion': fusion,
                                                                                'dataset': dataset, 'confidence': confidence,'target': target}
                                                            params = dict_merge([params, params_dict])
                                                            args_lst.append(params)

    print(args_lst)
    for p in args_lst:
        run(p)
        print('finished!!!')








