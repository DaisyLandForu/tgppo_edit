import random
import pyscipopt
import ecole.observation
import torch
import torch_geometric
import numpy as np
import gzip
import pickle
from pathlib import Path
import ecole

from observation import ExploreThenStrongBranch, TreeFeature
import os

from torch.utils.data import Dataset, DataLoader
import re
import json
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor

import pyscipopt as scip
from env import SCIPCollectEnv
import copy

def get_max_instance_index(base_dir):
    max_index = -1
    pattern = re.compile(r"instance_(\d+)")
    
    if os.path.exists(base_dir):
        for dir_name in os.listdir(base_dir):
            match = pattern.search(dir_name)
            if match:
                current_index = int(match.group(1))
                if current_index > max_index:
                    max_index = current_index
    return max_index

def process_instance(
    instance_name, seed, k,
    instance_path, episode_counter
):
    """处理单个实例的并行函数"""
    # 创建文件夹
    instance_dir = f"samples/{instance_name}/instance_{episode_counter}"
    os.makedirs(instance_dir, exist_ok=True)
    
    env = SCIPCollectEnv()
    exp_dict, collect_dict = env.run_episode(
        instance=instance_path,
        name=os.path.basename(instance_path).replace('.mps.gz', ''), # 实例名称
        explorer='random',
        expert='relpscost',
        k=k,
        state_dims = {
            'var_dim': 25,
            'node_dim': 8,
            'mip_dim': 53
        },
        scip_seed=seed,
        cutoff_value=None,
        scip_limits = {
            'node_limit': -1,
            'time_limit': 3600.,
        },
        scip_params={
            'heuristics': False,        # enable primal heuristics
            'cutoff': False,             # provide cutoff (value needs to be passed to the environment)
            'conflict_usesb': False,    # use SB conflict analysis
            'probing_bounds': False,    # use probing bounds identified during SB
            'checksol': False,          # check LP solutions found during strong branching with propagation
            'reevalage': 0,             # number of intermediate LPs solved to trigger reevaluation of SB value
        },
        verbose=False,                   # 是否打印
    )
    
    # 存入数据
    ff = open(f"{instance_dir}/data.pkl", 'wb')
    pickle.dump(collect_dict, ff)
    ff.close()

    
    # 保存实例信息
    with open(f"{instance_dir}/info.json", "w", encoding="utf-8") as f:
        json.dump({
            "instance_name": instance_path,
            "episode_counter": episode_counter,
            "sample_counter": len(collect_dict)
        }, f, indent=4)
    with open(f"{instance_dir}/exp_dict.json", "w", encoding="utf-8") as f:
        json.dump(exp_dict, f, indent=4)
        
    return episode_counter, len(collect_dict)

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='get_dataset')
    parser.add_argument('--instance_name',type=str, help='案例名称',default='miplib_mid')
    parser.add_argument('--instance_max_samples',type=int, help='strong branching数据集样本数',default=25)
    parser.add_argument('--seed',type=int, help='随机数种子',default=0)
    parser.add_argument('--k',type=int, help='探索几步再收集',default=10)

    args = parser.parse_args()

    init_instance_name = args.instance_name

    random.seed(args.seed)
    np.random.seed(args.seed)


    test_instances_name = [
        # path.name for path in Path("/home/data1/branch-search-trees-dataset/test_instances").glob("*.mps.gz")
        path.name for path in Path("/home/data1/TBranT-dataset/test_instances").glob("*.mps.gz")
    ]


    train_instances = [
        # str(path) for path in Path("/home/data1/branch-search-trees-dataset/train_instances").glob("*.mps.gz")
        str(path) for path in Path("/home/data1/TBranT-dataset/train_instances").glob("*.mps.gz") 
        if not (path.name in test_instances_name)
    ]

    # 打乱文件
    np.random.shuffle(train_instances)

    # 确定工作进程数
    num_workers = 16

    # 使用ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        results = []

        for seed in range(5):
            for k in [0, 1, 5, 10, 15]:

                instance_name = f"{init_instance_name}_s{seed}_k{k}"
                
                # 获取最大索引
                max_index = get_max_instance_index(f"samples/{instance_name}/")
                start_index = max_index + 1
                end_index = min(start_index + args.instance_max_samples, len(train_instances))
                
                # 提交所有任务
                for idx, instance_path in enumerate(train_instances[start_index:end_index], start=start_index):
                    future = executor.submit(
                        process_instance,
                        instance_name=instance_name,
                        seed=seed,
                        k=k,
                        instance_path=instance_path,
                        episode_counter=idx
                    )
                    futures.append(future)
                
    # 处理完成的任务
    for future in as_completed(futures):
        try:
            result = future.result()
            results.append(result)
            print(f"Completed episode {result[0]} with {result[1]} samples")
        except Exception as e:
            print(f"Error processing instance: {e}")

    print(f"Parallel data generation completed. Processed {len(results)} instances.")