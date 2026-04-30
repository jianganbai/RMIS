import os
import yaml
import time
import torch
import argparse
import subprocess
import numpy as np
import multiprocessing as mp

from tqdm import tqdm
from omegaconf import OmegaConf, DictConfig
from typing import List

from utils.util import set_seed
from uni_detect.task.train_test.info import TRAIN_TEST_DS_LIST

RMIS_DS = {
    'anomaly_detection': [f'dcase2{i}' for i in range(6)],
    'fault_diagnosis': [
        'idmt_air', 'idmt_engine', 'wt_plane_gearbox', 'mafaulda_sound',
        'mafaulda_vib', 'sdust_bearing', 'sdust_gear', 'umged_sound',
        'umged_vib', 'umged_vol', 'umged_cur', 'pu_vib', 'pu_cur'
    ]
}
RMIS_DS_LIST = []
for task_ds_list in RMIS_DS.values():
    RMIS_DS_LIST.extend(task_ds_list)

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

    overall_pbar.close()
    [p.close() for p in gpu_pbar.values()]


def get_exp_dir(ds: str, model_name: str, rel_exp_dir: str) -> str:
    for task, task_ds_list in RMIS_DS.items():
        if ds in task_ds_list:
            break
    exp_dir = os.path.join(f'exp/{model_name}/rmis/fix', rel_exp_dir, task, ds)
    return exp_dir


def runner(
    ds: str,
    model_conf: DictConfig,
    rel_exp_dir: str,
    seed: int,
    basic_conf: str
) -> None:
    if ds in RMIS_DS['anomaly_detection']:
        if ds.startswith('dcase'):
            ds_conf = OmegaConf.load(f'rmis/ds_conf/ad/{ds[5:]}/register.yaml')
    else:
        ds_conf = OmegaConf.load(f'rmis/ds_conf/fd/{ds}/register_fix.yaml')
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
    if ds in RMIS_DS['anomaly_detection']:
        run_args = [
            'python', '-m', 'runner.im.im', '--conf', conf_path,
            '--basic_conf', basic_conf, '--exp_dir', exp_dir,
            '--seed', str(seed)
        ]
    else:
        if ds in TRAIN_TEST_DS_LIST:
            run_args = [
                'python', '-m', 'runner.cls.eval_cls', '--conf', conf_path,
                '--basic_conf', basic_conf, '--exp_dir', exp_dir, '--seed', str(seed)
            ]
        else:
            run_args = [
                'python', '-m', 'runner.cls.cls_reg_nrun', '--conf', conf_path,
                '--basic_conf', basic_conf, '--exp_dir', exp_dir, '--seed', str(seed)
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


def rmis_stat(
    model_name: str,
    rel_exp_dir: str,
    out_dir: str
) -> None:
    rmis_dict = {}
    for ds in RMIS_DS_LIST:
        exp_dir = get_exp_dir(ds, model_name, rel_exp_dir)
        if ds.startswith('dcase'):
            with open(os.path.join(exp_dir, f'min/{ds}/knn.csv'), 'r') as fp:
                info = fp.readlines()
                score = float(info[-1].split(':')[-1].strip())
        elif ds in RMIS_DS['fault_diagnosis']:
            if ds in TRAIN_TEST_DS_LIST:
                with open(os.path.join(exp_dir, 'knn/acc.csv'), 'r') as fp:
                    info = fp.readlines()
                    assert info[-2].startswith('macro-acc')
                    score = float(info[-2].split(':')[-1].strip())
            else:
                with open(os.path.join(exp_dir, 'r_stat.csv'), 'r') as fp:
                    info = fp.readlines()
                    assert info[-1].startswith('macro-acc')
                    score = float(info[-1].split()[-2])
        rmis_dict[ds] = score

    task_dict = {}
    for task in RMIS_DS.keys():
        task_dict[task] = np.mean([rmis_dict[ds] for ds in RMIS_DS[task]]).item()
    task_dict['RMIS_score'] = np.mean([task_dict[task] for task in RMIS_DS]).item()

    with open(os.path.join(out_dir, 'rmis_score.csv'), 'w') as fp:
        fp.write('RMIS Benchmark\n')
        fp.write('-' * 30 + '\n')
        for task in RMIS_DS:
            for ds in RMIS_DS[task]:
                fp.write(f"{ds}: {rmis_dict[ds]:.2f}\n")
            fp.write('-' * 30 + '\n')
            fp.write(f"{task}: {task_dict[task]:.2f}\n")
            fp.write('-' * 30 + '\n')
        fp.write(f"RMIS_score: {task_dict['RMIS_score']:.2f}")


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

    # put large datasets at front
    ds_list = [ds for ds in RMIS_DS_LIST if ds.startswith('umged')] + \
        [ds for ds in RMIS_DS_LIST if not ds.startswith('umged')]
    model_name = model_conf['model_name']
    rmis_out_dir = os.path.join('exp', model_name, 'rmis/fix', args.rel_exp_dir)
    os.makedirs(rmis_out_dir, exist_ok=True)

    # for ds in tqdm(ds_list):
    #     runner(ds, model_conf, args.rel_exp_dir, args.seed)
    M = mp.Process(target=monitor, args=(ds_list, num_gpu))
    M.start()
    p = mp.Pool(num_gpu)
    for ds in ds_list:
        p.apply_async(
            func=runner,
            args=(ds, model_conf.copy(), args.rel_exp_dir, args.seed, args.basic_conf),
            error_callback=lambda e: print(e)
        )
        time.sleep(5)  # avoid transient high I/O
    p.close()
    p.join()
    M.join()

    print(f'Output dir: {rmis_out_dir}')
    error_ds_list = list(error_ds)
    if len(error_ds_list) == 0:
        rmis_stat(model_name, args.rel_exp_dir, rmis_out_dir)
