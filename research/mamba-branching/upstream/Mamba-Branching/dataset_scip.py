import gzip
import pickle
from pathlib import Path
from random import sample
from torch.utils.data import Dataset, DataLoader
import numpy as np 
import os
import pickle
from typing import List, Dict, Tuple
import torch
from torch.utils.data import Dataset, random_split
import h5py

class clip_dataset_h5(Dataset):
    def __init__(self, h5_file, max_len = 1000000, node_dim=8, mip_dim=53, var_dim=25):
        """
        :param h5_file: str, pathway to the data H5 file
        :param node_dim: int, dimension of node state
        :param mip_dim: int, dimension of mip state
        :param var_dim: int, dimension of variable state
        """
        super(clip_dataset_h5, self).__init__()

        self.n_data = max_len
        
        # load the h5 file
        self.h5_file = h5py.File(h5_file, 'r')
        
        if self.n_data > len(self.h5_file['dataset']):
            self.n_data = len(self.h5_file['dataset'])
        
        self.dataset = self.h5_file['dataset'][:self.n_data]

        # define the dimensions of each feature
        self.node_dim = node_dim
        self.mip_dim = mip_dim
        self.var_dim = var_dim

        # define the number of data points
        # self.n_data = len(self.h5_file['dataset'])
        

    def __getitem__(self, index):
        x = self.dataset[index]
        return [
            torch.LongTensor([x[0]]),
            torch.FloatTensor(x[1:1 + self.node_dim]),
            torch.FloatTensor(x[1 + self.node_dim:1 + self.node_dim + self.mip_dim]),
            torch.FloatTensor(x[1 + self.node_dim + self.mip_dim:].reshape(-1, self.var_dim))
        ]

    def __len__(self):
        return self.n_data 

def clip_collate(batch):
    actions, node_features, mip_features, cands_mats = zip(*batch)
    # cands_mat = [tensor.shape=(cans_num, 25),...]
    original_lens = [x.shape[0] for x in cands_mats]
    max_candidates = max(original_lens)
    
    # 填充 cands_mats 到 (batch_size, max_candidates, 25)
    padded_cands = torch.zeros(len(batch), max_candidates, 25, dtype=torch.float32)
    masks = torch.zeros(len(batch), max_candidates, dtype=torch.bool)
    
    for i, (mat, length) in enumerate(zip(cands_mats, original_lens)):
        padded_cands[i, :length] = mat
        masks[i, :length] = True  # 标记有效部分
    
    return (
        torch.hstack(actions),  # 填充actions
        torch.cat(
            [
                torch.vstack(node_features),
                torch.vstack(mip_features)
            ],
            dim = 1
        ),
        padded_cands,
        masks,
    )


class transformer_dataset_h5(Dataset):
    def __init__(self, h5_file, max_seq_length=50, node_dim=8, mip_dim=53, var_dim=25):
        """
        :param h5_file: str, pathway to the data H5 file
        :param node_dim: int, dimension of node state
        :param mip_dim: int, dimension of mip state
        :param var_dim: int, dimension of variable state
        """
        super(transformer_dataset_h5, self).__init__()

        # load the h5 file
        self.h5_file = h5py.File(h5_file, 'r')

        # define the dimensions of each feature
        self.node_dim = node_dim
        self.mip_dim = mip_dim
        self.var_dim = var_dim
        self.max_seq_length = max_seq_length

        # define the number of data points
        self.n_data = len(self.h5_file['dataset'])

    def __len__(self):
        return self.n_data 
    
    def __getitem__(self, index):
        sequence = self.h5_file['dataset'][f'seq_{index}']
        actions = []
        nodes = []
        mips = []
        vars = []
        for i in range(len(sequence)):
            x = sequence[f'arr_{i}']
            actions.append(torch.LongTensor([x[0]]))
            nodes.append(torch.FloatTensor(x[1:1 + self.node_dim]))
            mips.append(torch.FloatTensor(x[1 + self.node_dim:1 + self.node_dim + self.mip_dim]))
            vars.append(torch.FloatTensor(x[1 + self.node_dim + self.mip_dim:].reshape(-1, self.var_dim)))
        
        # shape = (seq_len, dim)
        nodes = torch.stack(nodes, dim=0)
        mips = torch.stack(mips, dim=0)
        states = torch.cat((nodes, mips), dim=1)
        actions = torch.hstack(actions)
                
        max_candidate_num = max(c.shape[0] for c in vars)
        masks = torch.zeros(len(vars), max_candidate_num, dtype=torch.bool)
        for i, c in enumerate(vars):
            # c.shape = (candidate_num, action_dim)
            if c.shape[0] < max_candidate_num:
                vars[i] = torch.cat([c, torch.zeros(max_candidate_num - c.shape[0], c.shape[1])], dim=0)
                masks[i, c.shape[0]:] = True
                
        # shape = (seq_len, max_candidate_num, action_dim)
        candidate_features = torch.stack(vars, dim=0)
        
    
        return {
            'actions': actions,                         # shape = (seq_len, )
            'states': states,                           # shape = (seq_len, 61)
            'candidate_features': candidate_features,   # shape = (seq_len, max_candidate_num, 25)
            'candidate_masks': masks,                    # shape = (seq_len, max_candidate_num)
        }
        
def pad_collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    自定义collate函数，用于处理变长序列的padding
    
    参数:
        batch: 一个batch的数据列表，每个元素是一个包含'states'和'actions'的字典
        
    返回:
        包含padded 'states', 'actions'和'lengths'的字典
    """
    # 获取batch中所有序列的长度
    lengths = [len(item['states']) for item in batch]
    max_len = max(lengths)
    
    # 获取batch中每个数据的candidate_num
    candidate_nums = [item['candidate_features'].shape[1] for item in batch]
    max_candidate_num = max(candidate_nums)

    # 获取state和action的维度
    state_dim = batch[0]['states'].shape[1]
    var_dim = batch[0]['candidate_features'].shape[-1]
    
    # 初始化padded张量
    padded_states = torch.zeros(len(batch), max_len, state_dim)
    padded_actions = torch.zeros(len(batch), max_len, dtype=torch.long)
    # 这个mask是用于序列的mask
    masks = torch.zeros(len(batch), max_len)

    # padded_candidate_features.shape = (batch_size, max_seq_len, max_candidate_num, action_dim)
    # padded_candidate_masks.shape = (batch_size, max_seq_len, max_candidate_num)
    padded_candidate_features = torch.zeros(len(batch), max_len, max_candidate_num, var_dim)
    # 这个mask是用于序列和candidates的mask
    padded_candidate_masks = torch.zeros(len(batch), max_len, max_candidate_num, dtype=torch.bool)
    
    # 填充数据并创建mask
    for i, item in enumerate(batch):
        seq_len = lengths[i]
        padded_states[i, :seq_len] = item['states']
        padded_actions[i, :seq_len] = item['actions']
        masks[i, :seq_len] = 1  # 有效位置为1

        candidate_num = candidate_nums[i]
        padded_candidate_features[i, :seq_len, :candidate_num] = item['candidate_features']
        padded_candidate_masks[i, :seq_len, :candidate_num] = item['candidate_masks']
    
    return {
        'states': padded_states,
        'actions': padded_actions,
        'lengths': torch.tensor(lengths),
        'masks': masks.bool(),  # 转换为布尔mask
        'candidate_features': padded_candidate_features,
        'candidate_masks': padded_candidate_masks,
    }


def get_data_loader_clip(args, data_type="less"):

    # # 限制数据集大小，训练clip只用100000个数据
    # sample_files = [str(path) for path in Path("samples/{}/".format(args.instance_name)).glob("instance_*/sample_*.pkl") ]
    # np.random.shuffle(sample_files)
    # sample_files = sample_files[:args.max_clip_samples]

    # train_files = sample_files[: int(0.8 * len(sample_files))]
    # valid_files = sample_files[int(0.8 * len(sample_files)) :]
    
    # train_dataset = ClipDataset(train_files)
    # valid_dataset = ClipDataset(valid_files)
    
    if data_type == "less" or data_type == "mid":
        train_dataset = clip_dataset_h5(
            f"h5_data_{data_type}/clip_train.h5",
            # "/home/data1/branch-search-trees-dataset/train.h5",
            args.max_clip_samples,
        )
        
        valid_dataset = clip_dataset_h5(
            f"h5_data_{data_type}/clip_valid.h5",
            # "/home/data1/branch-search-trees-dataset/val.h5",
            args.max_clip_samples,
        )
    else:
        train_dataset = clip_dataset_h5(
            "h5_data/clip_train.h5",
            # "/home/data1/branch-search-trees-dataset/train.h5",
            args.max_clip_samples,
        )
        
        valid_dataset = clip_dataset_h5(
            "h5_data/clip_valid.h5",
            # "/home/data1/branch-search-trees-dataset/val.h5",
            args.max_clip_samples,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.clip_batch_size,
        shuffle=True, 
        num_workers=8,           # 根据CPU核心数调整
        pin_memory=True,         # 加速GPU传输
        persistent_workers=True, # 保持工作进程活跃
        collate_fn=clip_collate
    )
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=args.clip_batch_size,
        shuffle=False,
        num_workers=8, pin_memory=True, persistent_workers=True,
        collate_fn=clip_collate
    )

    return train_loader, valid_loader

def get_data_loader_transformer(args, data_type="less"):
    # root_path = "samples/{}/".format(args.instance_name)
    # dataset = SequenceDataset(root_path, max_seq_length=args.max_seq_length)
    
    # # 划分数据集
    # train_size = int(0.8 * len(dataset))
    # val_size = len(dataset) - train_size

    # train_dataset, val_dataset = random_split(
    #     dataset, [train_size, val_size]
    # )
    
    if data_type == "less" or data_type == "mid":
        try:
            train_dataset = transformer_dataset_h5(
                f"h5_data_{data_type}/transformer_train_{args.max_seq_length}.h5",
            )
            
            val_dataset = transformer_dataset_h5(
                f"h5_data_{data_type}/transformer_valid_{args.max_seq_length}.h5",
            )
        except:
            train_dataset = transformer_dataset_h5(
                f"/home/data1/python_file/llm_branch/h5_data_{data_type}/transformer_train_{args.max_seq_length}.h5"
            )
            
            val_dataset = transformer_dataset_h5(
                f"/home/data1/python_file/llm_branch/h5_data_{data_type}/transformer_valid_{args.max_seq_length}.h5"
            )
    else:
        try:
            train_dataset = transformer_dataset_h5(
                f"h5_data/transformer_train_{args.max_seq_length}.h5",
            )
            
            val_dataset = transformer_dataset_h5(
                f"h5_data/transformer_valid_{args.max_seq_length}.h5",
            )
        except:
            train_dataset = transformer_dataset_h5(
                f"/home/data1/python_file/llm_branch/h5_data/transformer_train_{args.max_seq_length}.h5"
            )
            
            val_dataset = transformer_dataset_h5(
                f"/home/data1/python_file/llm_branch/h5_data/transformer_valid_{args.max_seq_length}.h5"
            )
            
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=pad_collate,
        num_workers=8, pin_memory=True, persistent_workers=True

    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=pad_collate,
        num_workers=8, pin_memory=True, persistent_workers=True
    )
    
    return train_loader, val_loader
    
