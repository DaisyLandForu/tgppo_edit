import torch
from network import TreeGateNet, GroupedMamba
import pyscipopt
import numpy as np
import torch.nn.functional as F
import random
from scipy.stats import gmean
import json
import argparse
import logging
import os 
from pathlib import Path
from env import ILEvalEnv, SeqEvalEnv
import pickle
import json
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
import multiprocessing as mp

def log_print(*args, **kwargs):
    message = ' '.join(map(str, args))
    logging.info(message)
# 将 print 函数替换为 log_print 函数
print = log_print

from collections import OrderedDict


def process_evaluate(SEED, instances_name, instance_file_path, clip_model, llm_model, LLM_NAME, cutoffs, args):
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    
    if args.is_use_llm:
        env = SeqEvalEnv(device=args.device, max_seq_len=args.max_seq_len, relpscost_times=args.relpscost_times)
    else:
        env = ILEvalEnv(device=args.device, relpscost_times=args.relpscost_times)
    
    exp_dict_list = []
    # for i, instance_file_path in enumerate(instances):
    name = instances_name
    
    eps = np.finfo(np.float32).eps.item()
    with torch.no_grad():
        if args.is_use_llm:
            exp_dict = env.run_episode(
                instance=instance_file_path,
                name=name,
                policy=clip_model,
                policy_name="TreeGate",
                llm=llm_model,
                llm_name=LLM_NAME,
                state_dims = {
                    'var_dim': 25,
                    'node_dim': 8,
                    'mip_dim': 53
                },
                scip_seed=SEED,
                cutoff_value=cutoffs[name],
                scip_limits={
                    'node_limit': -1,
                    'time_limit': args.time_limit,
                },
                scip_params={
                    'heuristics': False,        # enable primal heuristics
                    'cutoff': True,             # provide cutoff (value needs to be passed to the environment)
                    'conflict_usesb': False,    # use SB conflict analysis
                    'probing_bounds': False,    # use probing bounds identified during SB
                    'checksol': False,
                    'reevalage': 0,
                },
                verbose=True,
                is_init = args.is_init
            )
        else:
            exp_dict = env.run_episode(
                instance=instance_file_path,
                name=name,
                policy=clip_model,
                policy_name="TreeGate",
                state_dims = {
                    'var_dim': 25,
                    'node_dim': 8,
                    'mip_dim': 53
                },
                scip_seed=SEED,
                cutoff_value=cutoffs[name],
                scip_limits={
                    'node_limit': -1,
                    'time_limit': args.time_limit,
                },
                scip_params={
                    'heuristics': False,        # enable primal heuristics
                    'cutoff': True,             # provide cutoff (value needs to be passed to the environment)
                    'conflict_usesb': False,    # use SB conflict analysis
                    'probing_bounds': False,    # use probing bounds identified during SB
                    'checksol': False,
                    'reevalage': 0,
                },
                verbose=True,
                is_init=args.is_init
            )
        
    exp_dict_list.append(exp_dict)
    # 为了防止误操作，改成续写模式
    with open(f"logs_miplib_{args.data_type}_{args.run_id}_{args.evaluate_id}/{args.llm_name}/{name}_{SEED}.json", "a", encoding="utf-8") as f:
        json.dump(exp_dict, f, indent=4)
    
    # return 


def remove_module_prefix(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v  # 去掉 "module."
        else:
            new_state_dict[k] = v
    return new_state_dict




if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', type=int, default=100, help='run id')
    parser.add_argument('--evaluate_id', type=int, default=13, help='evaluate id')
    parser.add_argument('--gpu_id', type=int, default=1, help='gpu id')
    parser.add_argument('--nb_eval_instances', type=int, default=100, help='nb eval instances')
    parser.add_argument('--llm_name', type=str, default='mamba', help='llm name')
    parser.add_argument('--time_limit', type=int, default=3600, help='time limit')
    parser.add_argument('--max_seq_len', type=int, default=49, help='max seq len')
    parser.add_argument('--relpscost_times', type=int, default=1, help='relpscost times')
    parser.add_argument('--max_embed_ids', type=int, default=100, help='max embed ids')
    parser.add_argument('--is_finetune_clip', action='store_true', help='Whether clip is finetuned or not')
    parser.add_argument('--is_use_llm', action='store_true', help='Whether llm is used or not')
    parser.add_argument('--is_init', action='store_true', help='init the seq')
    parser.add_argument('--data_type', type=str, default="less", help='data type')
    parser.add_argument('--evaluate_type', type=str, default="test", help='evaluate type')
    parser.add_argument('--num_workers', type=int, default=8, help='num workers')
    parser.add_argument('--epoch_num', type=int, default=99, help='epoch num')


    args = parser.parse_args()

    run_id = args.run_id
    
    DEVICE = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    args.device = DEVICE
    NB_EVAL_INSTANCES = args.nb_eval_instances
    LLM_NAME = args.llm_name
    
    """
    导入模型
    """
    # 导入projection model
    save_path = f"./models/{run_id}"
    if LLM_NAME == "gpt2" or LLM_NAME == "distilgpt2" or LLM_NAME == "transformer" or LLM_NAME == "mamba":
        clip_model = TreeGateNet(
            infimum=8
        )
    else:
        if args.is_use_llm:

            raise ValueError("Invalid LLM name")

        else:
            print("Use TreeGateNet")
            clip_model = TreeGateNet(
                infimum=8
            )

    if args.is_finetune_clip:
        clip_model.load_state_dict(
            torch.load(
                "{}/model_finetune_{}.pth".format(save_path, args.epoch_num),
                map_location="cpu",
                weights_only=True
            )
        )
    else:
        clip_model.load_state_dict(
            torch.load(
                "{}/model_{}.pth".format(save_path, args.epoch_num),
                map_location="cpu",
                weights_only=True
            )
        )
    clip_model.to(DEVICE)
    clip_model.eval()

    if args.is_use_llm:
        if LLM_NAME == "transformer":
            from network import TransformerDecoder
            llm_model = TransformerDecoder(
                8, args.max_embed_ids
            )
            try:
                llm_model.load_state_dict(
                    torch.load(
                        f"./sequence_models/{run_id}/{LLM_NAME}_{args.epoch_num}.pth",
                        map_location="cpu",
                        weights_only=True
                    ),
                    strict=False
                )
            
            except:
                llm_model.load_state_dict(
                    remove_module_prefix(torch.load(
                        f"./sequence_models/{run_id}/{LLM_NAME}_{args.epoch_num}.pth",
                        map_location="cpu",
                        weights_only=True
                    ))
                )
            llm_model.to(DEVICE)
            
        elif LLM_NAME == "mamba":
            llm_model = GroupedMamba(
                8, args.max_embed_ids
            )
            try:
                llm_model.load_state_dict(
                    torch.load(
                        f"./sequence_models/{run_id}/mamba_{args.epoch_num}.pth",
                        map_location="cpu",
                        weights_only=True
                    )
                )
            
            except:
                llm_model.load_state_dict(
                    remove_module_prefix(torch.load(
                        f"./sequence_models/{run_id}/mamba_{args.epoch_num}.pth",
                        map_location="cpu",
                        weights_only=True
                    ))
                )
            llm_model.to(DEVICE)
    else:
        llm_model = None

    
    if not os.path.exists(f"logs_miplib_{args.data_type}_{args.run_id}_{args.evaluate_id}/{args.llm_name}"):
        os.makedirs(f"logs_miplib_{args.data_type}_{args.run_id}_{args.evaluate_id}/{args.llm_name}")
    
    logging.basicConfig(
        filename=f'./logs_miplib_{args.data_type}_{args.run_id}_{args.evaluate_id}/{args.llm_name}/results.log',
        level=logging.INFO
    )
    
    print("----------------------------------------------------------")
    print(f"run id: {run_id}")
    print(f"evaluate id: {args.evaluate_id}")
    print(f"gpu id: {args.gpu_id}")
    print(f"nb eval instances: {args.nb_eval_instances}")
    print(f"time limit: {args.time_limit}")
    print(f"max seq len: {args.max_seq_len}")
    print(f"max embed ids: {args.max_embed_ids}")
    print(f"is finetune clip: {args.is_finetune_clip}")
    print(f"is use llm: {args.is_use_llm}")
    print(f"is init: {args.is_init}")


    if args.data_type == "less":
        instances = [
                str(path) for path in Path(f"/home/data1/branch-search-trees-dataset/{args.evaluate_type}_instances").glob("*.mps.gz")
            ]
            
        instances_name = [
            path.name.split(".")[0] for path in Path(f"/home/data1/branch-search-trees-dataset/{args.evaluate_type}_instances").glob("*.mps.gz")
        ]
    elif args.data_type == "mid_easy":
        
        # with open("/home/data1/TBranT-dataset/test_instances/info.json", "r") as f:
        #     info_dict = json.load(f)
        
        # hard_instances_name = info_dict["hard"]
        # instances = [
        #     str(path) for path in Path("/home/data1/TBranT-dataset/test_instances").glob("*.mps.gz")
        #     if path.name.split(".")[0] not in hard_instances_name
        # ]
        
        # instances_name = [
        #     path.name.split(".")[0] for path in Path("/home/data1/TBranT-dataset/test_instances").glob("*.mps.gz")
        #     if path.name.split(".")[0] not in hard_instances_name
        # ]
        

        # hard_instances_name = info_dict["hard"]
        instances = [
            # str(path) for path in Path("/home/data1/TBranT-dataset-Own/test_instances_easy").glob("*.mps.gz")
            str(path) for path in Path("/home/data1/TBranT-dataset/test_instances").glob("*.mps.gz")
            # if path.name.split(".")[0] not in hard_instances_name
        ]
        
        instances_name = [
            # path.name.split(".")[0] for path in Path("/home/data1/TBranT-dataset-Own/test_instances_easy").glob("*.mps.gz")
            path.name.split(".")[0] for path in Path("/home/data1/TBranT-dataset/test_instances").glob("*.mps.gz")
            # if path.name.split(".")[0] not in hard_instances_name
        ]
        
    elif args.data_type == "mid_hard":
        with open("/home/data1/TBranT-dataset/test_instances/info.json", "r") as f:
            info_dict = json.load(f)
        
        hard_instances_name = info_dict["hard"]
        instances = [
            str(path) for path in Path("/home/data1/TBranT-dataset/test_instances").glob("*.mps.gz")
            if path.name.split(".")[0] in hard_instances_name
        ]
        
        instances_name = [
            path.name.split(".")[0] for path in Path("/home/data1/TBranT-dataset/test_instances").glob("*.mps.gz")
            if path.name.split(".")[0] in hard_instances_name
        ]
    
    elif args.data_type == "hard":
        # with open("/home/data1/benchmark/info.json", "r") as f:
        #     info_dict = json.load(f)
        
        # hard_instances_name = info_dict["hard"]
        
        # instances = [
        #     str(path) for path in Path("/home/data1/benchmark").glob("*.mps.gz")
        #     if path.name.split(".")[0] in hard_instances_name
        # ]
        
        # instances_name = [
        #     path.name.split(".")[0] for path in Path("/home/data1/benchmark").glob("*.mps.gz")
        #     if path.name.split(".")[0] in hard_instances_name
        # ]
        instances = [
            str(path) for path in Path("/home/data1/mamba-branching-hard").glob("*.mps.gz")
        ]
        instances_name = [
            path.name.split(".")[0] for path in Path("/home/data1/mamba-branching-hard").glob("*.mps.gz")
        ]
    
    else:
        raise ValueError("Invalid data type")
    
    if args.data_type == "less":
        cutoffs = pickle.load(open("/home/data1/branch-search-trees-dataset/cutoff_dict.pkl", 'rb'))

    elif args.data_type == "mid_easy" or args.data_type == "mid_hard":
        cutoffs = pickle.load(open("/home/data1/TBranT-dataset/cutoff_test.pkl", 'rb'))
    elif args.data_type == "hard":
        cutoffs = json.load(open("/home/data1/benchmark/cutoff.json", 'r'))
    else:
        raise ValueError("Invalid data type")
    
    mp.set_start_method('spawn')
    num_workers = args.num_workers

    # 使用ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        results = []
        
        for SEED in range(5):
            for i, instance_file_path in enumerate(instances):
                # if instances_name[i] == 'rail507':
                future = executor.submit(
                    process_evaluate,
                    SEED, 
                    instances_name[i], 
                    instance_file_path, 
                    clip_model, 
                    llm_model, 
                    LLM_NAME,
                    cutoffs, args
                )
                futures.append(future)
    # 处理完成的任务
    for future in as_completed(futures):
        try:
            result = future.result()
            results.append(result)
            # print(f"Completed episode {result[0]} with {result[1]} samples")
        except Exception as e:
            print(f"Error processing instance: {e}")

        # print("")
        # print("----------")
        # print(f"seed: {SEED}")
        
        
        # all_nnodes = []
        # all_fair_nnodes = []
        # for exp_dict in exp_dict_list:
        #     nnodes = exp_dict['nnodes']
        #     fair_nnodes = exp_dict['fair_nnodes']
        #     all_nnodes.append(nnodes)
        #     all_fair_nnodes.append(fair_nnodes)
        
        # print(f"nnodes: {gmean(all_nnodes)}, fair_nnodes: {gmean(all_fair_nnodes)}")

    