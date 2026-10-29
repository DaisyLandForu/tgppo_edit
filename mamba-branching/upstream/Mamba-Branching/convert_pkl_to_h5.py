""" Converter to transform collected pickle data files to IL inputs. """

import h5py
import numpy as np
import pickle
import argparse
import os
import glob
import torch
from sklearn.model_selection import train_test_split
from pathlib import Path

def convert_files(pkl_paths, args, type):    
    # create an h5 file
    print('Creating an h5 file...')
    if args.network_mode == 'clip':
        # dataset = f.create_dataset('dataset', (num_data,), dtype=dt)
        # counter = 0
        all_data = []
        for index, pkl in enumerate(pkl_paths):
            with open(pkl, 'rb') as f:
                print('\tProcessing {:s}, {:d} of {:d}...'.format(pkl.split('/')[-1], index + 1, len(pkl_paths)))
                D = pickle.load(f)
                for idx in D:
                    if D[idx]['cands_state_mat'].shape[0] > 1:  # We only collect data with more than 1 candidate
                        # flat_vector is always [target, node, mip, grid_flattened]
                        flat_vector = np.hstack([D[idx]['varRELpos'], D[idx]['node_state'], D[idx]['mip_state'],
                                                D[idx]['cands_state_mat'].flatten()]).astype('float32')
                        # dataset[counter] = flat_vector
                        # counter += 1
                        all_data.append(flat_vector)
        train_data = all_data
        f_train = h5py.File(os.path.join(args.out_dir, '{}_{}.h5'.format(args.network_mode, type)), 'w')
        dt = h5py.special_dtype(vlen=np.dtype('float32'))
        train_dataset = f_train.create_dataset('dataset', (len(train_data),), dtype=dt)
        
        for i in range(len(train_data)):
            train_dataset[i] = train_data[i]

        
        f_train.close()

    elif args.network_mode == 'transformer':
        
        all_data = []
        for index, pkl in enumerate(pkl_paths):
            with open(pkl, 'rb') as f:
                print('\tProcessing {:s}, {:d} of {:d}...'.format(pkl.split('/')[-1], index + 1, len(pkl_paths)))
                D = pickle.load(f)
                data_seq = []
                for idx in D:
                    if D[idx]['cands_state_mat'].shape[0] > 1:  # We only collect data with more than 1 candidate
                        flat_vector = np.hstack([D[idx]['varRELpos'], D[idx]['node_state'], D[idx]['mip_state'],
                                                D[idx]['cands_state_mat'].flatten()]).astype('float32')
                        data_seq.append(flat_vector)
                if len(data_seq) > 1 and len(data_seq) <= args.max_seq_length:
                    all_data.append(data_seq)
                
                elif len(data_seq) > args.max_seq_length:
                    result_sequences = []
                    total_length = len(data_seq)
                    num_subseq = (total_length + args.max_seq_length - 1) // args.max_seq_length  # 向上取整

                    for i in range(num_subseq):
                        start = i * args.max_seq_length
                        end = (i + 1) * args.max_seq_length
                        
                        sub_data_seq = data_seq[start:end]
                        result_sequences.append(sub_data_seq)
                    
                    all_data.extend(result_sequences)

        # train_data, valid_data = train_test_split(all_data, test_size=0.2, random_state=42)

        train_data = all_data
        f_train = h5py.File(os.path.join(args.out_dir, '{}_{}_{}.h5'.format(args.network_mode, type, args.max_seq_length)), 'w')
        # f_valid = h5py.File(os.path.join(args.out_dir, '{}_{}_{}.h5'.format(args.network_mode, "valid", args.max_seq_length)), 'w')
        
        train_group = f_train.create_group('dataset')
        # valid_group = f_valid.create_group('dataset')
        
        for i, seq in enumerate(train_data):# 为每个序列创建一个子组
            sub_group = train_group.create_group(f'seq_{i}')
            for j, arr in enumerate(seq):
                sub_group.create_dataset(f'arr_{j}', data=arr)
            sub_group.attrs['length'] = len(seq)
        
        f_train.close()

    print('finished!')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        '--pkl_file_dir',
        type=str,
        default='./samples/miplib_less'
    )

    parser.add_argument(
        '--out_dir',
        type=str,
        default='./h5_data_less/'
    )
    
    
    parser.add_argument(
        '--network_mode',
        type=str,
        default='transformer',
        help='clip or transformer'
    )
    
    parser.add_argument(
        '--max_seq_length',
        type=int,
        default=50,
        help='max sequence length'
    )
    
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # set the NumPy random seed
    np.random.seed(0)


    # stack all the state_vectors into a single matrix
    pkl_paths = []
    for i in range(4):
        pkl_paths.extend(sorted(glob.glob(args.pkl_file_dir + f'_s{i}_*/**/data.pkl',  recursive=True)))
        
    convert_files(pkl_paths, args, "train")

    pkl_paths = sorted(glob.glob(args.pkl_file_dir + '_s4_*/**/data.pkl',  recursive=True))
    
    convert_files(pkl_paths, args, "valid")
    

