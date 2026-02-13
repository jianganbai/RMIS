import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '5'  # TODO:

import gc
import copy
import torch
import ctypes
import argparse
import numpy as np
import pandas as pd
import multiprocessing as mp

from omegaconf import OmegaConf
from collections import defaultdict
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split

from utils.util import set_seed
from runner.emb.utt import extract_main
from uni_detect.task.emb_detect import EMB_KNN_Detection
from uni_detect.task.split.cal_acc import Split_Evaluator


def get_split_df(
    args: Dict[str, Any],
    seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(os.path.join(
        args['meta_data_dir'], f"{args['downstream']}/all.csv"
    ))
    train_csv = os.path.join(args['exp_dir'], 'train.csv')
    test_csv = os.path.join(args['exp_dir'], 'test.csv')

    # do not split one file into both train & test
    incre = 0
    if args.get('split_level', 'ori') == 'ori':  # default
        assert 'ori' in df.columns, 'Meta data should contain original file name'
        ori_map = defaultdict(list)
        for idx, row in df.iterrows():
            ori_map[row['ori']].append(idx)
        ori_list = list(ori_map.keys())
        lab_set = set(df[args.get('lab', 'scene')])
        while True:
            ori_train, ori_test = train_test_split(
                ori_list,
                test_size=args['test_size'],
                random_state=seed + incre
            )

            train_idx, test_idx = [], []
            for ori_tr in ori_train:
                train_idx.extend(ori_map[ori_tr])
            train_df = df.loc[train_idx].copy()
            tr_lab_set = set(train_df[args.get('lab', 'scene')])

            for ori_te in ori_test:
                test_idx.extend(ori_map[ori_te])
            test_df = df.loc[test_idx].copy()
            te_lab_set = set(test_df[args.get('lab', 'scene')])
            if tr_lab_set == lab_set and te_lab_set == lab_set:
                break
            else:
                incre += 100

        train_df.to_csv(train_csv)
        test_df.to_csv(test_csv)

    else:  # segment level split
        train_df, test_df = train_test_split(
            df,
            test_size=args['test_size'],
            random_state=args['seed']
        )
        train_df.to_csv(train_csv)
        test_df.to_csv(test_csv)

    return train_df, test_df


def split_runner(
    run_args: Dict[str, Any],
    shared_arr,
    run_id: int
) -> None:
    os.makedirs(run_args['exp_dir'], exist_ok=True)
    train_df, test_df = get_split_df(run_args, run_args['seed'] + run_id)
    reg_exp_dir = os.path.join(run_args['exp_dir'], 'knn')
    os.makedirs(reg_exp_dir, exist_ok=True)
    all_emb = np.frombuffer(
        shared_arr.get_obj(), dtype=np.float32
    ).reshape(len(train_df) + len(test_df), -1)

    EMB_KNN = EMB_KNN_Detection(
        args=run_args,
        exp_dir=reg_exp_dir,
    )
    # {file, label, pred} for only test
    pred_df = EMB_KNN.infer(
        emb_dict=None,
        exp_dir=reg_exp_dir,
        tr_emb=all_emb[train_df.index],
        te_emb=all_emb[test_df.index]
    )

    SE = Split_Evaluator(run_args)
    SE.cal_acc(pred_df, reg_exp_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default=None)
    parser.add_argument('--basic_conf', type=str, default='conf/basic.yaml')
    parser.add_argument('--exp_dir', type=str, default=None)
    parser.add_argument('--ntimes', type=int, help='number of runs', default=10)
    parser.add_argument('--emb_overwrite', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=111)
    args = parser.parse_args()
    set_seed(args.seed)

    ori_conf = OmegaConf.load(args.conf)
    basic_conf = OmegaConf.load(args.basic_conf)
    conf = OmegaConf.merge(ori_conf, basic_conf)
    conf = OmegaConf.to_container(conf, resolve=True)
    args = vars(args)
    args.update(conf)
    args.update(basic_conf)
    if args['downstream'] in ['idmt_engine']:  # no random split
        raise KeyError(f"Dataset {args['downstream']} does not require random split.")
    args['ntimes'] = max(args['ntimes'], 10)

    # extract embeddings at first and reuse them
    extract_dir = os.path.join(args['exp_dir'], 'emb')
    if not (os.path.exists(extract_dir) and len(os.listdir(extract_dir)) > 0) or \
            args['emb_overwrite']:
        extract_main({
            'conf': args['conf'],
            'basic_conf': args['basic_conf'],
            'extract_dir': extract_dir
        })
        gc.collect()

    # modify args for reusing pre-extracted embeds
    args['extract_dir'] = extract_dir
    args['infer_type'] = 'file'
    args['detect']['l2_norm'] = False  # pre-norm in main proc

    # read pre-saved emb_dict
    save_format = args.get('save_format', 'pt')
    data_file = os.path.join(
        args['extract_dir'], args['downstream'], f'all.{save_format}'
    )
    if save_format == 'npz':
        emb_dict = dict(np.load(data_file))
    else:
        # save numpy.ndarray in .pt
        emb_dict = torch.load(data_file, weights_only=False)

    # wrap emb data by shared array
    all_emb = np.stack(list(emb_dict.values()), dtype=np.float32)
    all_emb = all_emb / np.linalg.norm(all_emb, axis=1, keepdims=True)  # L2 normalization
    shared_arr = mp.Array(ctypes.c_float, all_emb.size)
    shared_emb = np.frombuffer(shared_arr.get_obj(), dtype=np.float32).reshape(all_emb.shape)
    shared_emb[:] = all_emb  # no copy

    # n-fold register
    # for run_id in range(args['ntimes']):
    #     run_args = copy.deepcopy(args)
    #     run_args['exp_dir'] = os.path.join(args['exp_dir'], f'r{run_id}')
    #     split_runner(run_args, shared_arr, run_id)
    P = []
    for run_id in range(args['ntimes']):
        run_args = copy.deepcopy(args)
        run_args['exp_dir'] = os.path.join(args['exp_dir'], f'r{run_id}')
        P.append(mp.Process(
            target=split_runner,
            args=(run_args, shared_arr, run_id),
        ))
    [p.start() for p in P]
    [p.join() for p in P]

    # post-process
    stat_cmd = (
        f"python -m utils.scripts.multirun_stat --ds {args['downstream']} "
        f"--exp_dir {args['exp_dir']} --prefix r --result_dir knn"
    )
    os.system(stat_cmd)
