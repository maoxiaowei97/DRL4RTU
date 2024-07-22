import numpy as np
from torch.utils.data import Dataset

class DRL4RTU_dataset(Dataset):
    def __init__(
            self,
            mode: str,
            params: dict, #parameters dict
    )->None:
        super().__init__()
        if mode not in ["train", "val", "test"]:
            raise ValueError
        path_key = {'train':'train_path', 'val':'val_path','test':'test_path'}[mode]
        path = params[path_key]
        self.data = np.load(path, allow_pickle=True).item()

    def __len__(self):
        return len(self.data['label_len'])

    def __getitem__(self, index):
        V = self.data['V'][index]
        pred_len = self.data['V_len'][index]
        V_reach_mask = self.data['V_reach_mask'][index]
        label = self.data['route_label'][index]
        label_len = self.data['label_len'][index]
        V_at = self.data['time_label'][index]
        start_fea = self.data['start_fea'][index]
        start_idx = self.data['start_idx'][index]
        route_label_all = self.data['route_label_all'][index]
        eta_label_len = self.data['eta_label_len'][index]
        first_node = self.data['first_node'][index]
        return V, V_reach_mask, label, label_len, V_at, start_fea, start_idx, eta_label_len, route_label_all, pred_len, first_node
