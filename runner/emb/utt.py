import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '2'  # TODO:

import glob
import torch
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm
from omegaconf import OmegaConf
from typing import Dict, Any
from collections import defaultdict
from torch.utils.data import (
    DataLoader,
)

from models.model_wrapper import construct_cls_model
from utils.util import (
    set_seed,
    ensemble_embs,
)
from datasets.build_dataset import read_data, get_dataset
from uni_detect.task.train_test.info import TRAIN_TEST_DS_LIST


class Worker:
    def __init__(
        self,
        args: Dict[str, Any],
        gpu_id: int
    ) -> None:
        self.args = args
        self.gpu_id = gpu_id
        self.DS = get_dataset(args)
        if args['downstream'] == 'dcase':
            self.ds = args['test_sets'][0]
        else:
            self.ds = args['downstream']
        self.save_format = args.get('save_format', 'pt')
        assert self.save_format in ['npz', 'pt'], f'Unsupported save format {self.save_format}'
        self.net = construct_cls_model(args)
        self.net.to(gpu_id)
        self.net.eval()

    def extract(self, csv) -> None:
        desc = os.path.splitext(os.path.relpath(
            csv,
            os.path.join(self.args['meta_data_dir'], self.ds)
        ))[0]
        save_path = os.path.join(
            self.args['extract_dir'],
            self.ds,
            desc + f'.{self.save_format}'
        )

        if self.args.get('use_shard', False) and \
            not self.ds.startswith('dcase') and \
                self.ds not in (['gwbw'] + TRAIN_TEST_DS_LIST):
            csv = pd.read_csv(csv)
            pre_read_data = read_data(self.args, csv)
        else:
            pre_read_data = None

        subset_ds = self.DS(
            ds=self.ds,
            csv=csv,
            args=self.args,
            hop_size=self.args['win_frames'],
            desc=desc,
            pre_read_data=pre_read_data
        )
        subset_dl = DataLoader(
            subset_ds,
            self.args.get('test_batch_size', self.args['batch_size'] * 4),
            shuffle=False,
            num_workers=self.args.get('num_workers', 4),
            pin_memory=False
        )

        seg_emb = []
        seg_file = []
        with torch.inference_mode():
            for medata in tqdm(subset_dl, desc=f'Extracting embeddings of {self.ds}'):
                x = medata['x'].cuda()
                with torch.autocast('cuda'):
                    output_dict = self.net(x=x, out_emb=True)
                emb = output_dict['embedding'].cpu().numpy()
                seg_emb.extend([emb[i] for i in range(emb.shape[0])])
                seg_file.extend(medata['file'])

        data = {}
        seg_map = defaultdict(list)
        for i, f in enumerate(seg_file):
            seg_map[f].append(i)
        for f in seg_map:
            data[f] = ensemble_embs(
                embs=np.stack([seg_emb[i] for i in seg_map[f]], axis=0),
                ensem_mode=self.args.get('feat_aggre', self.args['aggregation'])
            )

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if self.save_format == 'npz':
            np.savez(save_path, **data)
        elif self.save_format == 'pt':
            torch.save(data, save_path)
        else:
            raise KeyError(f'Unknown save format {self.save_format}')

    def release(self) -> None:
        self.net.cpu()
        del self.net
        torch.cuda.empty_cache()


def extract_main(args: Dict[str, Any]):
    conf = OmegaConf.load(args['conf'])
    basic_conf = OmegaConf.load(args['basic_conf'])
    conf = OmegaConf.merge(conf, basic_conf)
    conf = OmegaConf.to_container(conf, resolve=True)
    extract_dir = args['extract_dir']
    args.update(conf)
    if extract_dir is not None:
        args['extract_dir'] = extract_dir

    args['win_frames'] = int(args['input_duration'] * args['sample_rate'])

    if args['downstream'] == 'dcase':
        assert len(args['test_sets']) == 1
        all_csv_path = [
            file
            for path, _, _ in os.walk(os.path.join(args['meta_data_dir'], args['test_sets'][0]))
            for file in glob.glob(os.path.join(path, '*.csv'))
        ]
    else:
        all_csv_path = [
            file
            for path, _, _ in os.walk(os.path.join(args['meta_data_dir'], args['downstream']))
            for file in glob.glob(os.path.join(path, '*.csv'))
        ]

    w = Worker(args, 0)
    assert len(all_csv_path) > 0, 'No matched csv files!'
    for csv in all_csv_path:
        w.extract(csv)
    w.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default=None)
    parser.add_argument('--basic_conf', type=str, default='conf/basic.yaml')
    parser.add_argument('--extract_dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=111)
    args = parser.parse_args()
    set_seed(args.seed)
    args = vars(args)
    extract_main(args)
