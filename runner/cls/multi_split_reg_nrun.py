import os

os.environ['OMP_NUM_THREADS'] = '1'

import gc
import copy
import time
import torch
import queue
import ctypes
import argparse
import numpy as np
import multiprocessing as mp

from tqdm import tqdm
from scipy import integrate
from typing import List
from omegaconf import OmegaConf

from utils.util import set_seed
from runner.emb.utt import extract_main
from runner.cls.cls_reg_nrun import get_split_df
from uni_detect.task.emb_detect import EMB_KNN_Detection
from uni_detect.task.split.cal_acc import Split_Evaluator
from uni_detect.task.split.info import SPLIT_DS_MAP


task_queue = mp.Queue()


def multi_split_runner(shared_arr) -> None:
    while True:
        try:
            run_args, run_id = task_queue.get(timeout=5)
        except queue.Empty:
            continue
        if run_args is None or run_id is None:
            task_queue.put((None, None))
            return

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


def cal_split_auc(
    exp_dir: str,
    test_size_list: List[float],
    ds: str,
    method: str = 'simps'
) -> None:
    train_size_list = [0.0]
    acc_list = [100.0 / len(SPLIT_DS_MAP[ds])]
    for test_size in test_size_list[::-1]:  # invert
        train_size_list.append(1.0 - test_size)
        with open(os.path.join(exp_dir, str(test_size), 'r_stat.csv'), 'r') as fp:
            results = fp.readlines()
            acc_list.append(float(results[-1].split()[-2].strip()))
    train_size_list.append(1.0)
    acc_list.append(100.0)
    y, x = np.array(acc_list) / 100.0, np.array(train_size_list)

    if method == 'trapz':
        auc = integrate.trapezoid(y, x)
    elif method == 'simps':
        auc = integrate.simpson(y=y, x=x)
    with open(os.path.join(exp_dir, 'multi_split_auc.csv'), 'w') as fp:
        fp.write('train_size,acc\n')
        for i in range(1, len(test_size_list) + 1):
            fp.write(f"{train_size_list[i]:.2f},{acc_list[i]:.2f}\n")
        fp.write('-' * 30 + '\n')
        fp.write(f'AUC: {auc * 100.0:.2f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default=None)
    parser.add_argument('--basic_conf', type=str, default='conf/basic.yaml')
    parser.add_argument('--exp_dir', type=str, default=None)
    parser.add_argument('--ntimes', type=int, help='number of runs', default=10)
    parser.add_argument('--emb_overwrite', action='store_true', default=False)
    parser.add_argument('--nproc', type=int, default=10)
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

    # modify config
    ms_conf = args['multi_split_conf']
    test_size_list = [
        round(float(ts), 2) for ts in np.arange(
            ms_conf['min_test_size'],
            ms_conf['max_test_size'] + ms_conf['delta'],
            ms_conf['delta']
        )
    ]
    args['extract_dir'] = extract_dir
    args['infer_type'] = 'file'
    args['detect']['l2_norm'] = False
    del args['multi_split_conf']

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

    # iterate through test_size, n-fold register
    P = [
        mp.Process(target=multi_split_runner, args=(shared_arr,))
        for _ in range(args['nproc'])
    ]
    [p.start() for p in P]
    for test_size in test_size_list:
        sp_args = copy.deepcopy(args)
        # exp_dir ends with test_size, not train_size!
        sp_args['exp_dir'] = os.path.join(args['exp_dir'], str(test_size))
        sp_args['test_size'] = test_size
        del sp_args['nproc']

        # multiple runs
        for run_id in range(sp_args['ntimes']):
            run_args = copy.deepcopy(sp_args)
            run_args['exp_dir'] = os.path.join(run_args['exp_dir'], f'r{run_id}')
            task_queue.put((run_args, run_id))
    task_queue.put((None, None))
    num_task = len(test_size_list) * args['ntimes']
    pbar = tqdm(total=num_task, desc=f"Evaluating multi_split by {args['nproc']} workers")
    while True:
        qsize = task_queue.qsize()
        pbar.n = num_task + 1 - qsize
        pbar.refresh()
        if qsize == 1:
            break
        else:
            time.sleep(1)
    [p.join() for p in P]
    pbar.close()
    end = task_queue.get()
    assert end[0] is None and task_queue.qsize() == 0
    task_queue.close()

    # post-process
    for test_size in test_size_list:
        stat_cmd = (
            f"python -m utils.scripts.multirun_stat --ds {args['downstream']} "
            f"--exp_dir {args['exp_dir']}/{test_size} --result_dir knn --prefix r"
        )
        os.system(stat_cmd)
    cal_split_auc(args['exp_dir'], test_size_list, args['downstream'])
    print(f"Output dir: {args['exp_dir']}")
