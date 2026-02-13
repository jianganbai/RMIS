import os
import numpy as np
import pandas as pd

from typing import Dict, Any, Tuple

from .info import SPLIT_DS_MAP


class Split_Evaluator:
    '''
        Evaluator for manual train test split
    '''
    def __init__(
        self,
        args: Dict[Any, Any],
    ) -> None:
        self.ds = args['downstream']
        self.train_df = pd.read_csv(
            os.path.join(args['exp_dir'], 'train.csv'),
            index_col=0
        )
        self.test_df = pd.read_csv(
            os.path.join(args['exp_dir'], 'test.csv'),
            index_col=0
        )

        if 'umged' in self.ds:
            if args['lab'] == 'scene_E':
                self.CLS_MAP = SPLIT_DS_MAP['umged_E']
            elif args['lab'] == 'scene_G':
                self.CLS_MAP = SPLIT_DS_MAP['umged_G']
        elif self.ds in SPLIT_DS_MAP:
            self.CLS_MAP = SPLIT_DS_MAP[self.ds]
        else:
            train_lab_set = set(self.train_df[args.get('lab', 'scene')].unique())
            test_lab_set = set(self.test_df[args.get('lab', 'scene')].unique())
            all_lab_set = sorted(train_lab_set | test_lab_set)
            self.CLS_MAP = {lab: i for i, lab in enumerate(all_lab_set)}
        self.INV_CLS_MAP = {v: k for k, v in self.CLS_MAP.items()}

        self.train_df['cls'] = self.train_df.apply(
            lambda x: self.CLS_MAP[x[args.get('lab', 'scene')]],
            axis=1
        )
        self.test_df['cls'] = self.test_df.apply(
            lambda x: self.CLS_MAP[x[args.get('lab', 'scene')]],
            axis=1
        )

        self.train_map = {row['file']: row['cls'] for _, row in self.train_df.iterrows()}
        self.num_classes = len(self.CLS_MAP)
        self.cls_one_hot_lab = np.zeros((self.num_classes, self.num_classes), dtype=np.float32)
        for i in range(self.num_classes):
            self.cls_one_hot_lab[i, i] = 1

    def cal_acc(
        self,
        pred_df: pd.DataFrame,
        exp_dir: str
    ) -> Tuple[float, Dict[str, Any]]:
        cos_sim = []
        neb_cls = []
        for i in range(pred_df.shape[0]):
            item = pred_df.loc[i, 'pred']
            sim = []
            neb = []
            for x in item:
                sim.append(x[0])
                neb.append(self.train_map[x[1]])
            cos_sim.append(sim)
            neb_cls.append(neb)
        cos_sim = np.array(cos_sim)  # (num_samples, num_neb)
        neb_cls = np.array(neb_cls)  # (num_samples, num_neb)

        neb_one_hot = (
            self.cls_one_hot_lab[neb_cls.flatten()]
        ).reshape(cos_sim.shape[0], cos_sim.shape[1], self.num_classes)  # (num_samples, num_neb, num_cls)
        logits = np.matmul(cos_sim[:, np.newaxis, :], neb_one_hot).squeeze(1)
        pred_df['pred'] = logits.argmax(-1).tolist()

        pred_df.drop('label', axis=1, inplace=True)
        assert pred_df['file'].to_list() == self.test_df['file'].to_list()
        pred_df['cls'] = self.test_df['cls'].to_list()
        pred_df['eval'] = pred_df.apply(lambda x: int(x['cls'] == x['pred']), axis=1)

        result_df = pd.DataFrame({'scene': [], 'pred': [], 'correct': [], 'all': [], 'acc': []})
        num_acc, num_all = 0, 0
        for lab, lab_id in self.CLS_MAP.items():
            cls_bool = (pred_df['cls'] == lab_id)
            num_cls_cor = pred_df.loc[cls_bool, 'eval'].sum()
            num_cls = cls_bool.sum()
            result_df.loc[len(result_df)] = [
                lab,
                (pred_df['pred'] == lab_id).sum(),
                num_cls_cor,
                num_cls,
                num_cls_cor / num_cls * 100
            ]
            num_acc += num_cls_cor
            num_all += num_cls
        macro_acc = result_df['acc'].mean()
        micro_acc = num_acc / num_all * 100
        result_df['acc'] = result_df['acc'].map(lambda x: round(x, 2))

        pred_df['pred'] = pred_df['pred'].map(lambda x: self.INV_CLS_MAP[x])
        pred_df['cls'] = pred_df['cls'].map(lambda x: self.INV_CLS_MAP[x])
        pred_df.to_csv(os.path.join(exp_dir, 'pred.csv'), index=False)

        with open(os.path.join(exp_dir, 'acc.csv'), 'w') as fp:
            fp.write(result_df.__repr__() + '\n')
            fp.write('-' * 35 + '\n\n')
            fp.write(f'macro-acc: {macro_acc:.2f}\n')
            fp.write(f'micro-acc: {micro_acc:.2f}\n')
        return macro_acc, {'macro-acc': macro_acc}
