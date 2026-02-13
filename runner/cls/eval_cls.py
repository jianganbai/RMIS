import os
import copy
import torch
import pickle
import logging
import argparse
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from omegaconf import OmegaConf
from collections import defaultdict
from typing import Dict, Any, Tuple
from torch.utils.data import Dataset, DataLoader

from models.model_wrapper import construct_cls_model
from utils.util import (
    set_seed,
    ensemble_embs,
)
from datasets.build_dataset import (
    build_split_dataset,
    build_train_test_dataset,
)
from uni_detect.task.emb_detect import EMB_KNN_Detection
from uni_detect.task.split.cal_acc import Split_Evaluator
from uni_detect.task.train_test.cal_acc import Train_Test_Evaluator
from uni_detect.task.train_test.info import TRAIN_TEST_DS_LIST


def evaluate(
    net: nn.Module,
    train_ds: Dataset,
    test_ds: Dataset,
    args: Dict[str, Any],
    exp_dir: str
) -> Tuple[float, Dict[str, Any]]:
    net.eval()
    test_dl = DataLoader(
        test_ds,
        args.get('test_batch_size', args['batch_size'] * 4),
        shuffle=False,
        num_workers=4
    )
    os.makedirs(exp_dir, exist_ok=True)

    train_dl = DataLoader(
        train_ds,
        args.get('test_batch_size', args['batch_size'] * 4),
        shuffle=False,
        num_workers=args.get('num_workers', 4)
    )
    seg_emb = defaultdict(list)
    with torch.no_grad():
        for dt, dl in zip(['train', 'test'], [train_dl, test_dl]):
            for medata in tqdm(dl, desc=f'Extracting {dt} embeddings'):
                x = medata['x'].cuda()
                with torch.autocast('cuda'):
                    output_dict = net(x=x, out_emb=True)
                emb = output_dict['embedding'].cpu().numpy()
                for i in range(len(medata['file'])):
                    seg_emb[medata['file'][i]].append(emb[i])

    emb_dict = {}
    for file, seg in seg_emb.items():
        emb_dict[file] = ensemble_embs(
            embs=np.stack(seg, axis=0),
            ensem_mode=args.get('feat_aggre', args['aggregation'])
        )
    with open(os.path.join(exp_dir, 'emb.pkl'), 'wb') as fp:
        pickle.dump(emb_dict, fp)

    EMB_KNN = EMB_KNN_Detection(
        args=args,
        exp_dir=exp_dir,
    )
    # {file, label, pred} for only test
    pred_df = EMB_KNN.infer(
        emb_dict=emb_dict,
        exp_dir=exp_dir,
    )

    if args['downstream'] in ['idmt_engine']:
        args['map_'] = copy.deepcopy(EMB_KNN.label_fn.map_)
        TT = Train_Test_Evaluator(args)
        acc, info = TT.cal_acc(pred_df, exp_dir)
    else:  # default to split evaluator
        SE = Split_Evaluator(args)
        acc, info = SE.cal_acc(pred_df, exp_dir)
    return acc, info


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
    if args['downstream'] in TRAIN_TEST_DS_LIST:
        train_ds, test_ds = build_train_test_dataset(
            args=args,
            hop_size=hop_size,
        )
        args['infer_type'] = 'cls'
    else:  # default to manual split
        train_ds, test_ds = build_split_dataset(
            args=args,
            hop_size=hop_size,
        )
        args['infer_type'] = 'file'

    # network
    print('Setting up network...')
    net = construct_cls_model(args)
    net.cuda()
    logging.info(args)

    print(f'[Total: {sum(p.numel() for p in net.parameters()):.2e}]')
    evaluate(net, train_ds, test_ds, args, os.path.join(args['exp_dir'], 'knn'))

    if net is not None:
        net.cpu()
    del net
    del train_ds
    del test_ds


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
    args.update(basic_conf)

    main(args)
