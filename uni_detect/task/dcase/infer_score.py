import os
import yaml
import pandas as pd
import numpy as np

# from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Generator

from uni_detect.task.dcase.common import SET_MT_MAP
from uni_detect.detector.asd_methods import Anomaly_Detector


class DCASE_Anomaly_Detection:
    def __init__(
        self,
        dataset: str,
        conf_dir: str,
        meta_data_dir: str,
        score_train: bool = False,
        score_aggre: Optional[str] = None,
        overwrite_conf: Optional[Dict[str, Any]] = None
    ) -> None:
        conf_path = os.path.join(conf_dir, f'{dataset}.yaml')
        with open(conf_path, 'r', encoding='utf-8') as fp:
            conf = yaml.safe_load(fp)
        if overwrite_conf is not None:
            conf.update(overwrite_conf)

        self.dataset = dataset
        self.scale = conf['scale']
        self.domsp = conf['domsp']
        self.detector = conf['detector']
        self.mt_dict = SET_MT_MAP[dataset]

        self.score_train = score_train
        self.score_aggre = score_aggre

        self.check_args()
        for d in self.detector:
            if conf[f'{d}_conf'] is None:
                conf[f'{d}_conf'] = {}
            conf[f'{d}_conf']['score_train'] = score_train
        if self.domsp == 'min':
            self.AD = {
                0: Anomaly_Detector(conf, prefix='source'),
                1: Anomaly_Detector(conf, prefix='target')
            }
        else:
            self.AD = Anomaly_Detector(conf)

        self.read_meta_data(meta_data_dir)

    def check_args(self):
        assert self.scale in ['mt2set', 'mt', 'sec', 'id'], \
            f'Unknown scale {self.scale}'
        assert self.domsp in ['none', 'min'], \
            f'Unknown domain mode {self.domsp}'
        assert not (self.dataset in ['dcase20', 'dcase21'] and self.domsp in ['min']), \
            f'{self.dataset.upper()} dataset should not use domain generlization technique!'
        assert not (self.scale == 'mt2set' and self.dataset == 'dcase23'), \
            'The machine types are different for dev and eval in dcase23!'

    def read_meta_data(self, meta_data_dir: str):
        self.meta_data = {}
        if self.scale == 'mt2set':  # dev集和eval集合在一起进行检测
            self.meta_data['deval'] = {}
            for sett in ['train', 'test']:
                for mt in self.mt_dict['dev']:
                    self.meta_data['deval'][mt] = {}
                    dev_f = os.path.join(
                        meta_data_dir, self.dataset,
                        'dev', sett, f'{mt}.csv'
                    )
                    eval_f = os.path.join(
                        meta_data_dir, self.dataset,
                        'eval', sett, f'{mt}.csv'
                    )
                    dev_df = pd.read_csv(dev_f, sep=',')
                    eval_df = pd.read_csv(eval_f, sep=',')
                    self.meta_data['deval'][mt][sett] = pd.concat([dev_df, eval_df], axis=0)
        else:
            for setn in ['dev', 'eval']:
                self.meta_data[setn] = {}
                for mt in self.mt_dict[setn]:
                    self.meta_data[setn][mt] = {}
                    for sett in ['train', 'test']:
                        f = os.path.join(
                            meta_data_dir, self.dataset,
                            setn, sett, f'{mt}.csv'
                        )
                        self.meta_data[setn][mt][sett] = pd.read_csv(f, sep=',')
            if self.scale in ['sec', 'id']:
                self.secid_dict = {}
                for setn in ['dev', 'eval']:
                    self.secid_dict[setn] = {}
                    for mt in self.mt_dict[setn]:
                        tr_secid = set(self.meta_data[setn][mt]['train'][self.scale])
                        te_secid = set(self.meta_data[setn][mt]['test'][self.scale])
                        for s in te_secid:
                            assert s in tr_secid, \
                                f'For {mt} of {setn} set, {te_secid} should be fully included in {tr_secid}!'
                        self.secid_dict[setn][mt] = {'train': tr_secid, 'test': te_secid}

    def slice_sample(
        self
    ) -> Generator[Tuple[str, str, Dict[str, Dict[Any, Any]]], None, None]:
        # 默认每个机器单独检测，不slice源域/目标域
        if self.scale == 'mt2set':  # combine dev and eval for knn
            for mt in self.mt_dict['dev']:
                yield 'deval', mt, {'train': {}, 'test': {}}
        elif self.scale == 'mt':  # either dev or eval for knn
            for setn in ['dev', 'eval']:
                for mt in self.mt_dict[setn]:
                    yield setn, mt, {'train': {}, 'test': {}}
        elif self.scale in ['sec', 'id']:  # each sec/id for knn (default)
            for setn in ['dev', 'eval']:
                for mt in self.mt_dict[setn]:
                    for s in self.secid_dict[setn][mt]['test']:
                        args = {'train': {self.scale: s}, 'test': {self.scale: s}}
                        yield setn, mt, args
        else:
            raise KeyError(f'Unknown infer scale {self.scale}')

    def construct_score_dict(self):
        self.score_dict = {d: {} for d in self.detector}
        self.sett_list = ['test', 'train'] if self.score_train else ['test']
        for d in self.detector:
            for setn in ['dev', 'eval']:
                for mt in self.mt_dict[setn]:
                    for sett in self.sett_list:
                        key = f'{setn}_{mt}_{sett}'
                        self.score_dict[d][key] = None

    def aggre_score(
        self,
        file_list: List[str],
        score_array: np.ndarray
    ) -> np.ndarray:
        if self.score_aggre is None:
            return score_array
        else:
            df = pd.DataFrame({'file': file_list, 'score': score_array.tolist()})
            if self.score_aggre == 'mean':
                ndf = df.groupby('file').agg('mean')
            elif self.score_aggre == 'min':
                ndf = df.groupby('file').agg('min')
            elif self.score_aggre == 'max':
                ndf = df.groupby('file').agg('max')
            else:
                raise NotImplementedError(f'Score aggregation {self.score_aggre} is not implemented!')
            return np.array(ndf['score'])

    def score(
        self,
        emb_dict: Dict[str, np.ndarray],
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        '''score each file based on the embedding

        Args:
            emb_dict (Dict[str, np.ndarray]): {file: emb}

        Returns:
            {detector: {setn_mt_sett: pd.DataFrame}}
        '''
        self.file_table = {f: i for i, f in enumerate(emb_dict.keys())}
        self.construct_score_dict()
        if self.domsp == 'none':  # do not distinguish source and target
            score_dict = self.score_none(emb_dict)
        elif self.domsp == 'min':  # min decision
            score_dict = self.score_min(emb_dict)
        else:
            raise NotImplementedError(f'Unknown domsp strategy {self.domsp}')

        return score_dict

    def score_none(
        self,
        emb_dict: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        '''do not distinguish source and target

        Args:
            emb_dict (Dict[str, np.ndarray]): {file: emb}

        Returns:
            {detector: {setn_mt_sett: pd.DataFrame}}
        '''
        for setn, mt, args in self.slice_sample():
            df = self.meta_data[setn][mt]
            embs, file_slice = {}, {}
            set_file_list = {}
            for sett in ['train', 'test']:  # select embeddings
                if args[sett] != {}:
                    bools = np.all(
                        [df[sett][k] == v for k, v in args[sett].items()], axis=0
                    )
                else:
                    bools = [True for _ in range(df[sett].shape[0])]
                file_slice[sett] = df[sett][bools]
                set_emb_data = [emb_dict[f] for f in file_slice[sett]['file']]
                if set_emb_data[0].ndim <= 1:  # 1 file 1 emb
                    set_file_list[sett] = file_slice[sett]['file'].to_list()
                    if set_emb_data[0].ndim == 0:
                        embs[f'embs_{sett}'] = np.array(set_emb_data)[:, np.newaxis]
                    else:
                        embs[f'embs_{sett}'] = np.stack(set_emb_data)
                elif set_emb_data[0].ndim == 2:  # 1 file n emb
                    set_file_list[sett] = [
                        f
                        for i, f in enumerate(file_slice[sett]['file'])
                        for _ in range(set_emb_data[i].shape[0])
                    ]
                    embs[f'embs_{sett}'] = np.concatenate(set_emb_data, axis=0)

            # calculate anomaly score
            score_dict = self.AD.score(embs)

            # add meta_data to score_dict
            for d in self.detector:
                for sett in self.sett_list:
                    if self.dataset == 'dcase20':
                        ndf = file_slice[sett].drop(['path'], axis=1, inplace=False)
                    else:
                        ndf = file_slice[sett].drop(['attri', 'path'], axis=1, inplace=False)

                    ndf['score'] = self.aggre_score(set_file_list[sett], score_dict[d][sett])

                    key = f'{setn}_{mt}_{sett}'
                    if self.score_dict[d][key] is None:
                        self.score_dict[d][key] = ndf
                    else:  # multiple sections
                        self.score_dict[d][key] = pd.concat(
                            [self.score_dict[d][key], ndf], axis=0
                        )

        df_dict = {d: self.score_dict[d] for d in self.detector}
        return df_dict

    def score_min(
        self,
        emb_dict: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        '''two groups of detectors, one based on source, the other based on target

        Args:
            emb_dict (Dict[str, np.ndarray]): {file: emb}

        Returns:
            {detector: {setn_mt_sett: pd.DataFrame}}
        '''
        for setn, mt, args in self.slice_sample():
            df = self.meta_data[setn][mt]

            # select test embeddings and train info
            file_slice = {}
            for sett in ['train', 'test']:
                if args[sett] != {}:
                    bools = np.all(
                        [df[sett][k] == v for k, v in args[sett].items()], axis=0
                    )
                else:
                    bools = [True for _ in range(df[sett].shape[0])]
                file_slice[sett] = df[sett][bools]
            test_emb_data = [emb_dict[f] for f in file_slice['test']['file']]

            if test_emb_data[0].ndim <= 1:  # 1 file 1 emb
                test_file_list = file_slice['test']['file'].to_list()
                if test_emb_data[0].ndim == 0:
                    test_embs = np.array(test_emb_data)[:, np.newaxis]
                else:
                    test_embs = np.stack(test_emb_data)
            elif test_emb_data[0].ndim == 2:  # 1 file n emb
                test_file_list = [
                    f
                    for i, f in enumerate(file_slice['test']['file'])
                    for _ in range(test_emb_data[i].shape[0])
                ]
                test_embs = np.concatenate(test_emb_data)
            else:
                raise AttributeError(
                    f'Shape {test_emb_data[0].shape} of test embedding is incompatible!'
                )

            # select train embeddings
            train_args = {0: {'domain': 0}, 1: {'domain': 1}}
            if args['train'] != {}:
                for dom in [0, 1]:
                    train_args[dom].update(args['train'])
            train_embs = {}
            train_emb_ids = {}
            train_file_list = {}
            for dom in [0, 1]:
                bools = np.all(
                    [df['train'][k] == v for k, v in train_args[dom].items()], axis=0
                )
                dom_file_slice = df['train'][bools]
                train_emb_ids[dom] = list(dom_file_slice.index)
                train_emb_data = [emb_dict[f] for f in dom_file_slice['file']]
                if train_emb_data[0].ndim == 1:
                    train_file_list[dom] = dom_file_slice['file'].to_list()
                    train_embs[dom] = np.array(train_emb_data)

                elif train_emb_data[0].ndim == 2:
                    train_file_list[dom] = [
                        f
                        for i, f in enumerate(dom_file_slice['file'])
                        for _ in range(train_emb_data[i].shape[0])
                    ]
                    train_embs[dom] = np.concatenate(train_emb_data)

            # calculate anomaly score
            score_dict = {d: {sett: {} for sett in self.sett_list} for d in self.detector}
            sou_score_dict = self.AD[0].score({
                'embs_test': test_embs,
                'embs_train': train_embs[0],
                'embs_all_train': np.concatenate([train_embs[0], train_embs[1]])
            })
            tar_score_dict = self.AD[1].score({
                'embs_test': test_embs,
                'embs_train': train_embs[1],
                'embs_all_train': np.concatenate([train_embs[0], train_embs[1]])
            })
            for d in self.detector:
                for sett in self.sett_list:
                    if sett == 'train':  # use ground-truth detector
                        train_sou_score = self.aggre_score(
                            train_file_list[0],
                            sou_score_dict[d]['train']
                        )
                        train_tar_score = self.aggre_score(
                            train_file_list[1],
                            tar_score_dict[d]['train']
                        )
                    else:
                        score = np.min(np.concatenate([
                            sou_score_dict[d]['test'][np.newaxis, :],
                            tar_score_dict[d]['test'][np.newaxis, :]], axis=0), axis=0
                        )
                        score_dict[d]['test'] = {
                            'score_sou': sou_score_dict[d]['test'],
                            'score_tar': tar_score_dict[d]['test'],
                            'score': score
                        }

            # add meta_data to score_dict
            for d in self.detector:
                for sett in self.sett_list:
                    if self.dataset == 'dcase20':
                        ndf = file_slice[sett].drop(['path'], axis=1, inplace=False)
                    else:
                        ndf = file_slice[sett].drop(['attri', 'path'], axis=1, inplace=False)
                    if sett == 'test':
                        ndf['score_sou'] = self.aggre_score(
                            test_file_list,
                            score_dict[d][sett]['score_sou']
                        )
                        ndf['score_tar'] = self.aggre_score(
                            test_file_list,
                            score_dict[d][sett]['score_tar']
                        )
                        ndf['score'] = self.aggre_score(
                            test_file_list,
                            score_dict[d][sett]['score']
                        )

                    else:
                        ndf['score'][train_emb_ids[0]] = train_sou_score
                        ndf['score'][train_emb_ids[1]] = train_tar_score

                    key = f'{setn}_{mt}_{sett}'
                    if self.score_dict[d][key] is None:
                        self.score_dict[d][key] = ndf
                    else:
                        self.score_dict[d][key] = pd.concat(
                            [self.score_dict[d][key], ndf], axis=0)

        df_dict = {d: self.score_dict[d] for d in self.detector}
        return df_dict
