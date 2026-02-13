import os
import torch
import pickle
import logging
import argparse
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from omegaconf import OmegaConf
from typing import Dict, Any
from torch.utils.data import Dataset, DataLoader

from models.model_wrapper import construct_cls_model
from utils.util import (
    set_seed,
    ensemble_embs,
)
from datasets.build_dataset import build_dcase_dataset
from uni_detect.task.dcase.infer_score import DCASE_Anomaly_Detection
from uni_detect.task.dcase.cal_aucs import cal_aucs


def adall(
    net: nn.Module,
    test_ds: Dict[str, Dataset],
    args: Dict[str, Any],
    exp_dir: str
):
    net.eval()
    for k in test_ds.keys():
        emb_file = os.path.join(exp_dir, k, 'emb.pkl')
        if os.path.exists(emb_file):
            with open(emb_file, 'rb') as fp:
                emb_dict = pickle.load(fp)
        else:
            test_dl = DataLoader(
                test_ds[k],
                args.get('test_batch_size', args['batch_size'] * 4),
                shuffle=False,
                num_workers=args.get('num_workers', 4)
            )
            seg_emb = {'file': [], 'emb': []}
            with torch.no_grad():
                for medata in tqdm(test_dl, desc='Extracting embeddings'):
                    x = medata['x'].cuda()
                    # NOTE: amp is enabled for speed up! results will be different with no amp.
                    with torch.autocast('cuda'):
                        output_dict = net(
                            x=x,
                            out_emb=True
                        )
                    seg_emb['emb'].append(output_dict['embedding'].cpu().numpy())
                    seg_emb['file'].extend(medata['file'])
            seg_emb['emb'] = np.concatenate(seg_emb['emb'], axis=0)

            emb_dict = {}
            file_list = sorted(list(set(seg_emb['file'])))
            seg_map = {f: [] for f in file_list}
            for i, f in enumerate(seg_emb['file']):
                seg_map[f].append(i)
            for f in file_list:
                emb_dict[f] = ensemble_embs(
                    embs=seg_emb['emb'][seg_map[f]],
                    ensem_mode=args.get('feat_aggre', args['aggregation'])
                )

            os.makedirs(os.path.join(exp_dir, k), exist_ok=True)
            with open(emb_file, 'wb') as fp:
                pickle.dump(emb_dict, fp)

        DCASE_AD = DCASE_Anomaly_Detection(
            dataset=k,
            conf_dir=args['AD_conf_dir'],
            meta_data_dir=args['meta_data_dir'],
            score_aggre=args.get('score_aggre', None),
            overwrite_conf=args.get('detect', None)
        )
        score_dict = DCASE_AD.score(emb_dict)
        cal_aucs(
            score_dict=score_dict,
            dataset=k,
            exp_dir=exp_dir
        )


def main(args: Dict[str, Any]) -> None:
    # exp dir
    os.makedirs(args['exp_dir'], exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(args['exp_dir'], 'log'),
        format='%(asctime)s  %(levelname)s %(message)s',
        level=logging.INFO,
        force=True  # remove previous handlers
    )

    # data
    args['win_frames'] = int(args['input_duration'] * args['sample_rate'])
    hop_dur = args.get('hop_duration', None)
    if hop_dur is None:
        hop_size = args['win_frames']
    else:
        hop_size = int(hop_dur * args['sample_rate'])

    print('Loading data...')
    test_ds, _ = build_dcase_dataset(
        args=args,
        datasets=args['test_sets'],
        ds_mt={d: [] for d in args['test_sets']},
        hop_size=hop_size,
        cond=[],
    )

    # network
    print('Setting up network...')
    net = construct_cls_model(args)
    net.cuda()

    print(f'[Total: {sum(p.numel() for p in net.parameters()):.2e}]')
    adall(net, test_ds, args, os.path.join(args['exp_dir'], 'min'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default=None)
    parser.add_argument('--basic_conf', type=str, default='conf/basic.yaml')
    parser.add_argument('--exp_dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=111)
    args = parser.parse_args()
    set_seed(args.seed)

    conf = OmegaConf.load(args.conf)
    basic_conf = OmegaConf.load(args.basic_conf)
    conf = OmegaConf.merge(conf, basic_conf)
    conf = OmegaConf.to_container(conf, resolve=True)
    args = vars(args)
    args.update(conf)

    os.makedirs(args['exp_dir'], exist_ok=True)
    main(args)
