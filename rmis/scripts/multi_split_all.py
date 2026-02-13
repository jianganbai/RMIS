import os
import yaml
import time
import torch
import argparse
import subprocess
import pandas as pd
import multiprocessing as mp

from tqdm import tqdm
from omegaconf import OmegaConf, DictConfig
from typing import List

from utils.util import set_seed
from rmis.scripts.reg_all import RMIS_DS
from uni_detect.task.train_test.info import TRAIN_TEST_DS_LIST


MS_DS_LIST = [ds for ds in RMIS_DS['fault_diagnosis'] if ds not in TRAIN_TEST_DS_LIST]
manager = mp.Manager()
gpu_queue = manager.Queue()
status_queue = manager.Queue()
error_ds = manager.list()


def monitor(ds_list: List[str], num_gpu: int):
    ds_queue = set(ds_list)
    error_list = []
    num_finish = 0
    overall_pbar = tqdm(
        desc=f'Evaluating on {num_gpu} GPUs for {len(ds_list)} datasets',
        total=len(ds_list),
        position=0
    )

    gpu_pbar = {
        gpu_id: tqdm(desc=f'GPU {gpu_id}: idle', position=i+1)
        for i, gpu_id in enumerate(gpu_list)
    }
    gpu_status = {
        gpu_id: {'Succeed': [], 'Fail': []}
        for gpu_id in gpu_list
    }

    while num_finish < len(ds_list):
        gpu_id, ds, status = status_queue.get()
        if status == 'processing':
            gpu_pbar[gpu_id].set_description(f'GPU {gpu_id}: processing {ds}')
        elif status in ['succeed', 'fail']:
            ds_queue.discard(ds)
            num_finish += 1
            overall_pbar.update(1)
            if status == 'fail':
                error_list.append(ds)

            gpu_status[gpu_id][status.capitalize()].append(ds)
            gpu_pbar[gpu_id].update(1)
            gpu_pbar[gpu_id].set_description(f'GPU {gpu_id}: idle')
            gpu_pbar[gpu_id].set_postfix(gpu_status[gpu_id])


def get_exp_dir(ds: str, model_name: str, rel_exp_dir: str) -> str:
    for task, task_ds_list in RMIS_DS.items():
        if ds in task_ds_list:
            break
    exp_dir = os.path.join(f'exp/{model_name}/rmis/multi_split', rel_exp_dir, task, ds)
    return exp_dir


def runner(
    ds: str,
    model_conf: DictConfig,
    rel_exp_dir: str,
    seed: int,
    nproc: int,
    basic_conf: str
) -> None:
    ds_conf = OmegaConf.load(f'rmis/ds_conf/fd/{ds}/register_multi_split.yaml')
    conf = OmegaConf.merge(ds_conf, model_conf)
    if 'ds_spcf' in model_conf:
        if ds in model_conf['ds_spcf']:
            conf = OmegaConf.merge(conf, model_conf['ds_spcf'][ds])
        del conf['ds_spcf']
    conf = OmegaConf.to_container(conf, resolve=True)

    exp_dir = get_exp_dir(ds, conf['model_name'], rel_exp_dir)
    os.makedirs(exp_dir, exist_ok=True)
    conf_path = os.path.join(exp_dir, 'conf.yaml')
    with open(conf_path, 'w', encoding='utf-8') as fout:
        yaml.dump(conf, fout, allow_unicode=True, sort_keys=False)

    gpu_id = gpu_queue.get()
    status_queue.put((gpu_id, ds, 'processing'))
    sub_env = os.environ.copy()
    sub_env['CUDA_VISIBLE_DEVICES'] = gpu_id
    sub_env['OMP_NUM_THREADS'] = '1'  # constrain unnecessary cpu usage

    run_args = [
        'python', '-m', 'runner.cls.multi_split_reg_nrun', '--conf', conf_path,
        '--basic_conf', basic_conf, '--exp_dir', exp_dir, '--seed', str(seed),
        '--nproc', str(nproc)
    ]
    with open(os.path.join(exp_dir, 'run.log'), 'w') as flog:
        proc = subprocess.Popen(
            run_args,
            env=sub_env,
            stdout=flog,
            stderr=flog
        )
        proc.wait()
    gpu_queue.put(gpu_id)

    if proc.returncode != 0:
        error_ds.append(ds)
        status_queue.put((gpu_id, ds, 'fail'))
    else:
        status_queue.put((gpu_id, ds, 'succeed'))


def rmis_ms_stat(
    model_name: str,
    rel_exp_dir: str,
    out_dir: str
) -> None:
    rmis_ms_info = []
    for ds in MS_DS_LIST:
        exp_dir = get_exp_dir(ds, model_name, rel_exp_dir)
        with open(os.path.join(exp_dir, 'multi_split_auc.csv'), 'r') as fp:
            info = fp.readlines()
            assert info[-1].startswith('AUC')
            auc = float(info[-1].split()[-1].strip())
        rmis_ms_info.append(pd.Series({'Dataset': ds, 'AUC': auc}))
    df = pd.DataFrame(rmis_ms_info)
    mean_auc = df['AUC'].mean()

    with open(os.path.join(out_dir, 'rmis_ms_score.csv'), 'w') as fp:
        fp.write('RMIS Multi-Split Results\n')
        fp.write('-' * 30 + '\n')
        fp.write(df.__repr__() + '\n')
        fp.write('-' * 30 + '\n')
        fp.write(f"Mean AUC: {mean_auc:.2f}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_conf', type=str, default='rmis/model_conf/fisher/small.yaml')
    parser.add_argument('--basic_conf', type=str, default='conf/basic.yaml')
    parser.add_argument('--rel_exp_dir', type=str, default='small')
    parser.add_argument(
        '--gpu', type=str, default=','.join([str(i) for i in range(torch.cuda.device_count())])
    )
    parser.add_argument('--seed', type=int, default=111)
    args = parser.parse_args()
    set_seed(args.seed)
    model_conf = OmegaConf.load(args.model_conf)

    gpu_list = args.gpu.split(',')
    num_gpu = len(gpu_list)
    for i in gpu_list:
        gpu_queue.put(i)
    nproc = (mp.cpu_count() - 10) // num_gpu

    # put large datasets at front
    ds_list = [ds for ds in MS_DS_LIST if ds.startswith('umged')] + \
        [ds for ds in MS_DS_LIST if not ds.startswith('umged')]
    model_name = model_conf['model_name']
    rmis_out_dir = os.path.join('exp', model_name, 'rmis/multi_split', args.rel_exp_dir)
    os.makedirs(rmis_out_dir, exist_ok=True)

    # for ds in tqdm(ds_list):
    #     runner(ds, model_conf, args.rel_exp_dir, args.seed)
    M = mp.Process(target=monitor, args=(ds_list, num_gpu))
    M.start()
    p = mp.Pool(num_gpu)
    res = []
    for ds in ds_list:
        res.append(p.apply_async(
            func=runner,
            args=(ds, model_conf.copy(), args.rel_exp_dir, args.seed, nproc, args.basic_conf),
            error_callback=lambda e: print(e)
        ))
        time.sleep(5)  # avoid transient high I/O
    p.close()
    p.join()
    M.join()

    print(f'Output dir: {rmis_out_dir}')
    error_ds_list = list(error_ds)
    if len(error_ds_list) == 0:
        rmis_ms_stat(model_name, args.rel_exp_dir, rmis_out_dir)
