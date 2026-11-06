import json
from pathlib import Path
from scipy.stats.mstats import gmean
import numpy as np
import pickle as pkl
import glob
logs_path = 'logs_miplib_small'

# method_name = 'TreeGate-p'              # 3179.55 3179.55
# method_name = 'TreeGate'                # 2146.75 2180.04
# method_name = 'Mamba-Branching-p'       # 2272.43 2272.43
method_name = 'Mamba-Branching'         # 2068.83 2091.54
# method_name = 'Mamba-Branching-p-ncl'   # 3000.92 3000.92
# method_name = 'Transformer-Branching'   # 3078.56 3120.04
# method_name = 'Transformer-Branching-p' # 5138.15 5138.15
# method_name = 'T-BranT'

# logs_path = 'logs_miplib_large'
# method_name = 'easy-Mamba-Branching'    # 1819.32 2053.91
# # method_name = 'easy-T-BranT'
# # method_name = 'easy-TreeGate'    
# # method_name = 'easy-scip'    
# # method_name = 'hard-Mamba-Branching'    # 12319.55
# # method_name = 'hard-scip'  
# # method_name = 'hard-T-BranT'  
# # method_name = 'hard-TreeGate'  


all_fair_nnodes_list = []
all_pd_integral_list = []
all_nnodes_list = []
all_time_list = []
all_gap_list = []
for seed in range(5):
    
    info_seed_list = [
        str(path) for path  in Path(
            f"./{logs_path}/{method_name}"
        ).glob(f'*_{seed}.json')
    ]    
    if 'scip' in method_name:
        info_seed_list = [
            str(path) for path  in glob.glob(
                f"./{logs_path}/{method_name}"
                f'/SCIPEval_sandbox_{seed}_relpscost/*_{seed}_*.pkl'
            )
        ]
        data_list = []
        for info in info_seed_list:
            with open(info, 'rb') as f:
                data = pkl.load(f)
            data_list.append(data)
            print(f"name: {data['name']} seed:{data['seed']} fair_nnodes:{data['fair_nnodes']} time:{data['opt_time_process']}")
    
    else:
        if info_seed_list != []:
            
            data_list = []
            for info in info_seed_list:
                with open(info, 'r') as f:
                    data = json.load(f)
                data_list.append(data)
                print(f"name: {data['name']} seed:{data['seed']} fair_nnodes:{data['fair_nnodes']} time:{data['opt_time_process']}")

        else:
            info_seed_list = [
                str(path) for path  in Path(
                    f"./{logs_path}/{method_name}"
                ).glob(f'*_{seed}_ILEval_info.pkl')
            ]
            
            data_list = []
            for info in info_seed_list:
                with open(info, 'rb') as f:
                    data = pkl.load(f)
                data_list.append(data)
                print(f"name: {data['name']} seed:{data['seed']} fair_nnodes:{data['fair_nnodes']} time:{data['opt_time_process']}")
    
    fair_nnodes_list = []
    pd_integral_list = []
    nnodes_list = []
    time_list = []
    gap_list = []
    for data in data_list:
        fair_nnodes_list.append(data['fair_nnodes'])
        pd_integral_list.append(data['primaldualintegral']) 
        nnodes_list.append(data['nnodes'])
        time_list.append(data['opt_time_process'] if "T-BranT" not in method_name else data['scip_solve_time'])
        gap_list.append(data['gap'])

    all_fair_nnodes_list.extend(fair_nnodes_list)
    all_pd_integral_list.extend(pd_integral_list)
    all_nnodes_list.extend(nnodes_list)
    all_time_list.extend(time_list)
    all_gap_list.extend(gap_list)
    
    print("")
    print(f"instance num: {len(fair_nnodes_list)}")
    print(f"seed:{seed} gmean fair_nnodes: {gmean(fair_nnodes_list)} pd integral: {gmean(pd_integral_list)} time: {gmean(time_list)}")
    print("")

print(f'Overall nnodes: {gmean(all_nnodes_list) :.2f}')
print(f"Overall fair_nnodes: {gmean(all_fair_nnodes_list) :.2f}")
print(f"Overall pd integral: {gmean(all_pd_integral_list) :.2f}")
print(f"Overall time: {gmean(all_time_list) :.2f}")
print(f"Overall 1-shift time: {gmean([t+1 for t in all_time_list])-1 :.2f}")
print(f"Mean time: {np.mean(all_time_list) :.2f}")
print(f"Overall gap: {gmean([gap for gap in all_gap_list ]) :.2f}")
