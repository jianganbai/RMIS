import os
import glob
import torch
import pandas as pd

from collections import defaultdict
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, ConcatDataset
from typing import List, Dict, Any, Tuple, Optional

from .common import ALL_MT_MAP
from .audio_dataset import (
    read_data,
    Audio_Dataset,
    Audio_DatasetNG
)


def get_dataset(args: Dict[str, Any]):
    ds_type = args.get('ds_type', 'raw')
    if ds_type == 'ng':
        return Audio_DatasetNG
    elif ds_type == 'raw':
        return Audio_Dataset
    else:
        raise KeyError(f'Unknown dataset type {ds_type}')


class DCASEConcatDataset(ConcatDataset):
    def __init__(self, datasets: List[Dataset]):
        super().__init__(datasets)
        if hasattr(datasets[0], 'concat_sync_var'):
            self.concat_sync_var = datasets[0].concat_sync_var
            for varn in datasets[0].concat_sync_var:
                setattr(self, varn, getattr(datasets[0], varn))

        self.seg_weights = []
        for ds in self.datasets:
            if hasattr(ds, 'seg_weights'):
                self.seg_weights.extend(ds.seg_weights)


def build_dcase_dataset(
    args: Dict[str, Any],
    datasets: List[str],
    ds_mt: Dict[str, List[str]],
    hop_size: Optional[int],
    cond: List[str] = [],
    pre_read_data: Optional[Dict[str, List[torch.Tensor]]] = None,
    save_read_data: bool = False
) -> Tuple[Dict[str, Dataset], Dict[str, List[torch.Tensor]]]:
    meta_data_dir = args['meta_data_dir']
    all_datasets = {}
    read_data = {}
    DS = get_dataset(args)

    for ds in datasets:
        assert os.path.isdir(os.path.join(meta_data_dir, ds)), \
            f'No meta_data for {ds} in {meta_data_dir} directory'
        for c in ds_mt[ds]:  # select certain machines
            cond.append(c)

        # DCASE25
        # if ds == 'dcase25':  # TODO:
        # if ds == 'dcase25' and 'train' not in cond:  # TODO:
        #     cond.append('dev')

        all_csv_path = [
            file
            for path, _, _ in os.walk(os.path.join(meta_data_dir, ds))
            for file in glob.glob(os.path.join(path, '*.csv'))
        ]
        all_csv_path = [f for f in all_csv_path if f.split('/')[-2] in ['train', 'test']]
        for c in cond:
            all_csv_path = [f for f in all_csv_path if c in f]
        all_csv_path = sorted(
            all_csv_path,
            key=lambda x: 0 if 'dev/' in x else 1
        )
        all_csv_path = sorted(
            all_csv_path,
            key=lambda x: ALL_MT_MAP[ds][os.path.basename(x).split('.')[0]]
        )

        ds_list = []
        for csv in all_csv_path:
            desc = os.path.splitext(os.path.relpath(
                csv,
                os.path.join(args['meta_data_dir'], ds)
            ))[0]
            if pre_read_data is not None and csv in pre_read_data.keys():
                audio_ds = DS(
                    ds=ds,
                    csv=csv,
                    args=args,
                    hop_size=hop_size,
                    desc=desc,
                    pre_read_data=pre_read_data[csv]
                )
            else:
                audio_ds = DS(
                    ds=ds,
                    csv=csv,
                    args=args,
                    hop_size=hop_size,
                    desc=desc
                )
            ds_list.append(audio_ds)
            if save_read_data:
                if hasattr(audio_ds, 'data'):
                    read_data[csv] = audio_ds.data
                else:
                    read_data[csv] = None
        all_datasets[ds] = DCASEConcatDataset(ds_list)

    # post processing after concatenation
    return all_datasets, read_data


def build_split_dataset(
    args: Dict[str, Any],
    hop_size: int,
) -> Tuple[Dataset, Dataset]:
    ds = args['downstream']
    assert os.path.isdir(os.path.join(args['meta_data_dir'], ds)), \
        f"No meta_data for {ds} in {args['meta_data_dir']} directory"

    df = pd.read_csv(os.path.join(args['meta_data_dir'], f'{ds}/all.csv'))
    train_csv = os.path.join(args['exp_dir'], 'train.csv')
    test_csv = os.path.join(args['exp_dir'], 'test.csv')
    if os.path.exists(train_csv):
        train_df = pd.read_csv(train_csv, index_col=0)
        test_df = pd.read_csv(test_csv, index_col=0)
    else:
        # do not split one file into both train & test
        run_id = 0
        if args.get('split_level', 'ori') == 'ori':
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
                    random_state=args['seed'] + run_id
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
                    run_id += 100

            train_df.to_csv(train_csv)
            test_df.to_csv(test_csv)

        else:  # segment level split
            train_df, test_df = train_test_split(
                df,
                test_size=args['test_size'],
                random_state=args['seed'] + run_id
            )
            train_df.to_csv(train_csv)
            test_df.to_csv(test_csv)

    # read shard globally
    if args.get('use_shard', False):
        data = read_data(args, df)
        pre_read_data = {
            'train': [data[i] for i in train_df.index.to_list()],
            'test': [data[j] for j in test_df.index.to_list()]
        }
    else:
        pre_read_data = {'train': None, 'test': None}
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    DS = get_dataset(args)
    train_ds = DS(
        ds=ds,
        csv=train_df,
        args=args,
        hop_size=hop_size,
        desc='train',
        pre_read_data=pre_read_data['train']
    )
    test_ds = DS(
        ds=ds,
        csv=test_df,
        args=args,
        hop_size=hop_size,
        desc='test',
        pre_read_data=pre_read_data['test']
    )
    return train_ds, test_ds


def build_train_test_dataset(
    args: Dict[Any, Any],
    hop_size: Optional[int],
) -> Tuple[Dataset, Dataset]:
    assert os.path.isdir(os.path.join(args['meta_data_dir'], args['downstream'])), \
        f"No meta_data for {args['downstream']} in {args['meta_data_dir']} directory"

    DS = get_dataset(args)
    train_ds = DS(
        ds=args['downstream'],
        csv=os.path.join(args['meta_data_dir'], args['downstream'], 'train.csv'),
        args=args,
        hop_size=hop_size,
        desc='train'
    )
    test_ds = DS(
        ds=args['downstream'],
        csv=os.path.join(args['meta_data_dir'], args['downstream'], 'test.csv'),
        args=args,
        hop_size=hop_size,
        desc='test'
    )
    return train_ds, test_ds
