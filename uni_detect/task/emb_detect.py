import os
import yaml
import faiss
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.special import softmax
from torchmetrics import Accuracy
from typing import Dict, Any, Optional
from matplotlib.ticker import MultipleLocator

from .train_test.info import TRAIN_TEST_DS_LIST
from datasets.label_fn import Key_Labeler


class EMB_KNN_Detection:
    def __init__(
        self,
        args: Dict[str, Any],
        exp_dir: str,
    ) -> None:
        conf_path = os.path.join(args['detect_conf_dir'], f"{args['downstream']}.yaml")
        overwrite_conf = args.get('detect', None)
        if os.path.exists(conf_path):
            with open(conf_path, 'r') as fp:
                conf = yaml.safe_load(fp)
        else:
            conf = {}
            assert overwrite_conf is not None, \
                f"No detection conf found for dataset {args['downstream']}!"
        if overwrite_conf is not None:
            conf.update(overwrite_conf)
        self.return_type = args['infer_type']
        self.fast_eval = args['infer_type'] == 'cls' and args.get('eval_test', True)

        # knn
        self.k = conf['k']
        self.min_k = conf.get('min_k', self.k)
        self.max_k = conf.get('max_k', self.k)
        self.l2_norm = conf.get('l2_norm', True)
        self.over_sample_conf = conf.get('over_sample', None)
        assert self.min_k <= self.k and self.k <= self.max_k, \
            f'Target k={self.k} should be within range of [{self.min_k}, {self.max_k}]'

        if args['downstream'] in TRAIN_TEST_DS_LIST:
            self.train_df = pd.read_csv(
                os.path.join(args['meta_data_dir'], args['downstream'], 'train.csv')
            )
            self.test_df = pd.read_csv(
                os.path.join(args['meta_data_dir'], args['downstream'], 'test.csv')
            )
            self.label_fn = Key_Labeler(args.get('lab', 'scene'))
        else:  # split dataset
            self.train_df = pd.read_csv(
                os.path.join(os.path.dirname(exp_dir), 'train.csv'),
                index_col=0
            )
            self.test_df = pd.read_csv(
                os.path.join(os.path.dirname(exp_dir), 'test.csv'),
                index_col=0
            )
            self.label_fn = Key_Labeler(args.get('lab', 'scene'))
        self.train_df['label'] = self.label_fn(self.train_df)
        self.test_df['label'] = self.label_fn(self.test_df)

    def infer(
        self,
        emb_dict: Optional[Dict[str, np.ndarray]],
        exp_dir: str,
        tr_emb: Optional[np.ndarray] = None,
        te_emb: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        '''Infer the test labels based on train labels

        Args:
            emb_dict (Dict[str, np.ndarray]): {file: emb}
            exp_dir (str): directory to plot acc-k curve

        Returns:
            pd.DataFrame: {file, label, pred}
        '''
        # knn
        tr_file = self.train_df['file'].to_list()
        tr_lab = self.train_df['label'].to_numpy()
        if emb_dict is not None:
            tr_emb = np.stack([emb_dict[f] for f in self.train_df['file']], axis=0)
            te_emb = np.stack([emb_dict[f] for f in self.test_df['file']], axis=0)
        else:
            assert tr_emb is not None and tr_emb.shape[0] == len(self.train_df)
            assert te_emb is not None and te_emb.shape[0] == len(self.test_df)
        if tr_emb.dtype != np.float32:
            tr_emb = tr_emb.astype(np.float32)
            te_emb = te_emb.astype(np.float32)
        if self.l2_norm:
            faiss.normalize_L2(tr_emb)
            faiss.normalize_L2(te_emb)
        index = faiss.IndexFlatL2(tr_emb.shape[-1])
        index.add(tr_emb)
        neb_dist, neb_id = index.search(te_emb, self.max_k)

        # construct one-hot
        num_class = np.unique(tr_lab).shape[0]
        one_hot_lab = np.zeros((tr_emb.shape[0], num_class), dtype=np.float32)
        for i in range(tr_emb.shape[0]):
            one_hot_lab[i, tr_lab[i]] = 1

        # construct meta data df for test
        pred_df = self.test_df[['file', 'label']].copy()
        pred_df.reset_index(drop=True, inplace=True)

        acc_dict = {}
        for k in range(self.min_k, self.max_k + 1):
            cos_sim = 1 - neb_dist[:, :k] / 2  # [B, k]
            neb_one_hot = (one_hot_lab[neb_id[:, :k].flatten()]).reshape(-1, k, num_class)  # [B, k, C]
            logits = np.matmul(cos_sim[:, np.newaxis, :], neb_one_hot).squeeze(1)  # [B, C]
            if self.return_type == 'prob':  # prob of each class
                pred_lab = softmax(logits, axis=-1).tolist()
            elif self.return_type == 'file':  # sim and file of neighbor. used for split ds
                pred_lab = []
                for i in range(cos_sim.shape[0]):
                    neb_info = []
                    for j in range(cos_sim.shape[1]):
                        neb_info.append((cos_sim[i, j], tr_file[neb_id[i, j]]))
                    pred_lab.append(neb_info)
            else:  # most similar class. used for train-test split
                pred_lab = logits.argmax(-1)

            if self.return_type == 'cls' and self.fast_eval:
                acc_metric = Accuracy(
                    task='multiclass',
                    num_classes=num_class,
                    average='macro'
                )
                acc_metric(torch.tensor(pred_lab), torch.tensor(pred_df['label'].to_numpy()))
                acc = float(acc_metric.compute())
                acc_dict[k] = acc * 100

            if k == self.k:
                k_pred_df = pred_df.copy()
                k_pred_df['pred'] = pred_lab

        if self.return_type == 'cls' and self.fast_eval:
            self.plot_k_acc(acc_dict, exp_dir)
        return k_pred_df

    def plot_k_acc(
        self,
        acc_dict: Dict[int, float],
        exp_dir: str
    ):
        plt.plot(list(acc_dict.keys()), list(acc_dict.values()))
        plt.xlabel('k')
        plt.ylabel('macro-acc')
        plt.title('KNN Detection Results')
        plt.grid(True)
        plt.xlim(self.min_k - 1, self.max_k + 1)
        ax = plt.gca()
        ax.xaxis.set_major_locator(MultipleLocator(2))
        plt.savefig(os.path.join(exp_dir, 'knn.png'), bbox_inches='tight')
        plt.close()
