import os
import copy
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Dict, Optional, Tuple
from scipy.stats import hmean
from sklearn.metrics import roc_auc_score

from uni_detect.task.dcase.common import SET_MT_MAP


def cal_aucs(
    score_dict: Dict[str, Dict[str, pd.DataFrame]],
    dataset: str,
    exp_dir: Optional[str],
    score_vis: bool = False,
    gen_sub: bool = False
):
    # score_dict: {detector: {setn_mt_sett: DataFrame}}
    if exp_dir is not None:
        if not exp_dir.endswith(dataset):
            exp_dir = os.path.join(exp_dir, dataset)
        os.makedirs(exp_dir, exist_ok=True)

    if dataset == 'dcase20':
        result, total = cal_aucs_dcase20(score_dict, exp_dir, score_vis=score_vis)
    elif dataset == 'dcase21':
        result, total = cal_aucs_dcase21(score_dict, exp_dir, score_vis=score_vis)
    elif dataset == 'dcase22':
        result, total = cal_aucs_dcase22(score_dict, exp_dir, score_vis=score_vis)
    elif dataset in ['dcase23', 'dcase24', 'dcase25']:
        result, total = cal_aucs_dcase232425(score_dict, exp_dir, dataset, score_vis=score_vis)
    elif dataset == 'dcase25':  # NOTE: deprecated
        result, total = cal_aucs_dcase25(score_dict, exp_dir, score_vis=score_vis)
    else:
        raise NotImplementedError(f'Validation on {dataset} not implemented!')

    if gen_sub:
        assert exp_dir is not None, 'exp_dir not specified!'
        if dataset in [f'dcase2{i}' for i in range(2, 6)]:
            gen_sub_files(score_dict, exp_dir)
        else:
            warnings.warn(f'Submission files for {dataset} dataset are not supported.')

    return result, total


def parse_scores(label, score):
    grouped_scores = [[], []]
    for la, s in zip(label, score):
        grouped_scores[la].append(s)
    # assert len(grouped_scores[0]) == len(grouped_scores[1]), \
    #     'Number of normal samples and anomalous samples do not equal!'
    return grouped_scores


def vis_score(scores, result, setn, out_dir):
    '''Draw score distribution

    Args:
        scores: {mt: {sec: [[normal scores], [anomaly scores]]}} or {mt: {sec: [scores]}}
        result: {mt: np.ndarray} or np.ndarray
        setn (str): set name
        out_dir (str): output directory
    '''
    mt_num = len(scores)
    max_sec_num = max([len(s) for s in scores.values()])
    for a in scores:
        for b in scores[a]:
            sec_item = len(scores[a][b])
            break
    fig = plt.figure(figsize=(5 * max_sec_num * (sec_item // 2), 3 * mt_num))
    ax = fig.subplots(mt_num, max_sec_num * (sec_item // 2))
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    for mt_id, mt in enumerate(scores.keys()):
        for sec_id, sec in enumerate(scores[mt].keys()):
            if len(scores[mt][sec]) == 4:  # sou-nor, sou-ano, tar-nor, tar-ano
                for i, dom in zip([0, 1], ['sou', 'tar']):
                    bins = np.linspace(min([min(scores[mt][sec][i * 2 + j]) for j in range(2)]),
                                       max([max(scores[mt][sec][i * 2 + j]) for j in range(2)]), 50)
                    ax[mt_id, sec_id * 2 + i].hist(scores[mt][sec][i * 2], bins=bins, alpha=0.3, label=f'{dom}-nor')
                    ax[mt_id, sec_id * 2 + i].hist(scores[mt][sec][i * 2 + 1], bins=bins, alpha=0.3, label=f'{dom}-ano')
                    pfm = '/'.join(map(lambda x: str(x), result[mt][sec_id]))
                    ax[mt_id, sec_id * 2 + i].set_title(f'{mt} - {sec} - {dom}: {pfm}')
                    ax[mt_id, sec_id * 2 + i].legend(['nor', 'ano'])
            elif len(scores[mt][sec]) == 2:  # nor, ano
                bins = np.linspace(min([min(scores[mt][sec][i]) for i in range(2)]),
                                   max([max(scores[mt][sec][i]) for i in range(2)]), 50)
                ax[mt_id, sec_id].hist(scores[mt][sec][0], bins=bins, alpha=0.3, label='nor')
                ax[mt_id, sec_id].hist(scores[mt][sec][1], bins=bins, alpha=0.3, label='ano')
                ax[mt_id, sec_id].set_title(f'{mt} - sec {sec}: {result[mt][sec_id, -1]:.2f}')
                ax[mt_id, sec_id].legend(['nor', 'ano'])
            elif len(scores[mt][sec]) == 1:  # unlabeled
                bins = np.linspace(min(scores[mt][sec][0]), max(scores[mt][sec][0]), 50)
                ax[mt_id, sec_id].hist(scores[mt][sec][0], bins=bins)
                ax[mt_id, sec_id].set_title(f'{mt} - sec {sec}')
            else:
                raise ValueError('scores dict value error!')
    # plt.suptitle(f"{'/'.join(dirn.split('/')[1:])}_{self.best_hm * 100:.2f}")
    fig.savefig(os.path.join(out_dir, f'{setn}_score_dist.png'), bbox_inches='tight')
    plt.close()


def plot_roc_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    save_path: str
):
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(save_path, bbox_inches='tight')


def gen_sub_files(
    score_dict: Dict[str, Dict[str, pd.DataFrame]],
    exp_dir: str
) -> None:
    '''Generate submission files (anomaly score & decision result)

    Args:
        score_dict (Dict[str, Dict[str, pd.DataFrame]]): {detector: {setn_mt_sett: DataFrame}}
        exp_dir (str): step_500/dcase24/
    '''
    ds = os.path.basename(exp_dir)
    assert ds in [f'dcase2{i}' for i in range(2, 6)], \
        'Only support submission format of dcase22 - dcase25'
    for d in score_dict.keys():  # detector
        os.makedirs(os.path.join(exp_dir, d), exist_ok=True)
        for setn_mt_sett, df in score_dict[d].items():
            setn, mt, sett = setn_mt_sett.split('_')
            if sett == 'train':
                continue

            # anomaly score
            for sec, sdf in df.groupby('sec', sort=True):
                score_table = pd.DataFrame({
                    'file': sdf['file'].apply(lambda x: os.path.basename(x)),
                    'score': sdf['score']
                })
                score_csv = os.path.join(exp_dir, d, f'anomaly_score_{mt}_section_{sec:02d}_{sett}.csv')
                score_table.to_csv(score_csv, header=False, index=False)

            # decision result
            thres = np.median(score_table['score'].to_numpy())
            score_table['decision'] = score_table['score'].apply(lambda x: 1 if x > thres else 0)
            decision_csv = os.path.join(exp_dir, d, f'decision_result_{mt}_section_{sec:02d}_{sett}.csv')
            score_table[['file', 'decision']].to_csv(decision_csv, header=False, index=False)


def cal_aucs_dcase20(
    score_dict: Dict[str, Dict[str, pd.DataFrame]],
    exp_dir: Optional[str],
    score_vis: bool = True
):
    # score_dict: {detector: {setn_mt_sett: DataFrame}}
    result, unisec, scores = {}, {}, {}
    for d in score_dict.keys():  # detector
        result[d], unisec[d], scores[d] = {}, {}, {}
        for setn in ['dev', 'eval']:
            result[d][setn], unisec[d][setn], scores[d][setn] = {}, {}, {}
            for mt in SET_MT_MAP['dcase20'][setn]:
                result[d][setn][mt] = []
                df = score_dict[d][f'{setn}_{mt}_test']
                unisec[d][setn][mt] = np.unique(df['sec'])
                scores[d][setn][mt] = {}
                for sec in unisec[d][setn][mt]:
                    df_sec = df.loc[df['sec'] == sec]
                    # DCASE20 benchmark
                    auc = roc_auc_score(df_sec['status'], df_sec['score']) * 100
                    pauc = roc_auc_score(df_sec['status'], df_sec['score'], max_fpr=0.1) * 100
                    mean = np.array([auc, pauc]).mean()
                    result[d][setn][mt].append([auc, pauc, mean])
                    scores[d][setn][mt][sec] = parse_scores(df_sec['status'], df_sec['score'])
                mt_result = np.array(result[d][setn][mt])
                result[d][setn][mt].append([mt_result[:, i].mean() for i in range(3)])
                result[d][setn][mt] = np.array(result[d][setn][mt]).round(2)

    keys = ['auc', 'pauc', 'mean']
    total_mean = {}
    for d in score_dict.keys():
        subset_mean = {}
        if exp_dir is None:
            for setn in ['dev', 'eval']:
                subset_mean[setn] = float(np.array([
                    result[d][setn][mt][-1, -1]
                    for mt in SET_MT_MAP['dcase20'][setn]
                ]).mean())
        else:
            with open(os.path.join(exp_dir, f'{d}.csv'), 'w') as fp:
                for setn in ['dev', 'eval']:
                    mt_mean = []
                    fp.write('-' * 80 + '\n')
                    fp.write(f'{setn.upper()}\n\n')
                    for mt in SET_MT_MAP['dcase20'][setn]:
                        info1 = {'sec': unisec[d][setn][mt].tolist() + ['mean']}
                        info1.update({
                            keys[i]: result[d][setn][mt][:, i].tolist()
                            for i in range(3)
                        })
                        df1 = pd.DataFrame(info1)
                        mt_mean.append(result[d][setn][mt][-1, -1])
                        fp.write(f'{setn}_{mt}: {result[d][setn][mt][-1, -1]:.2f}\n')
                        fp.write(df1.to_csv(index=False))
                        fp.write('\n')
                    subset_mean[setn] = float(np.array(mt_mean).mean())
                    fp.write(f'subset_mean: {subset_mean[setn]:.2f}\n\n')
                    if score_vis:
                        vis_score(
                            scores=scores[d][setn],
                            result=result[d][setn],
                            setn=setn,
                            out_dir=exp_dir
                        )
                fp.write('-' * 80 + '\n')
                fp.write(f"total_mean: {(subset_mean['dev'] + subset_mean['eval']) / 2:.2f}\n")
        total_mean[d] = copy.deepcopy(subset_mean)

    return result, total_mean


def cal_aucs_dcase21(
    score_dict: Dict[str, Dict[str, pd.DataFrame]],
    exp_dir: Optional[str],
    score_vis: bool = True
):
    # score_dict: {detector: {setn_mt_sett: DataFrame}}
    result, unisec, scores = {}, {}, {}
    for d in score_dict.keys():
        result[d], unisec[d], scores[d] = {}, {}, {}
        for setn in ['dev', 'eval']:
            result[d][setn], unisec[d][setn] = {}, {}
            scores[d][setn] = {}
            for mt in SET_MT_MAP['dcase21'][setn]:
                result[d][setn][mt], scores[d][setn][mt] = [], {}
                df = score_dict[d][f'{setn}_{mt}_test']
                unisec[d][setn][mt] = np.unique(df['sec'])
                for sec in unisec[d][setn][mt]:
                    df_sec = df.loc[df['sec'] == sec]
                    # DCASE21 benchmark
                    df_sou = df_sec.loc[df['domain'] == 0]
                    df_tar = df_sec.loc[df['domain'] == 1]
                    auc_s = roc_auc_score(df_sou['status'], df_sou['score']) * 100
                    pauc_s = roc_auc_score(df_sou['status'], df_sou['score'], max_fpr=0.1) * 100
                    auc_t = roc_auc_score(df_tar['status'], df_tar['score']) * 100
                    pauc_t = roc_auc_score(df_tar['status'], df_tar['score'], max_fpr=0.1) * 100
                    hm = hmean([auc_s, pauc_s, auc_t, pauc_t])
                    result[d][setn][mt].append([auc_s, pauc_s, auc_t, pauc_t, hm])
                    scores[d][setn][mt][sec] = parse_scores(df_sou['status'], df_tar['score'])
                    scores[d][setn][mt][sec].extend(parse_scores(df_tar['status'], df_tar['score']))

                mt_result = np.array(result[d][setn][mt])
                result[d][setn][mt].append([hmean(mt_result[:, i]) for i in range(5)])
                result[d][setn][mt] = np.array(result[d][setn][mt]).round(2)

    keys = ['auc_s', 'pauc_s', 'auc_t', 'pauc_t', 'hmean']
    total_hm = {}
    for d in score_dict.keys():
        subset_hm = {}
        if exp_dir is None:
            for setn in ['dev', 'eval']:
                subset_hm[setn] = float(hmean([
                    result[d][setn][mt][-1, -1]
                    for mt in SET_MT_MAP['dcase21'][setn]
                ]))
                if score_vis:
                    vis_score(
                        scores=scores[d][setn],
                        result=result[d][setn],
                        setn=setn,
                        out_dir=exp_dir
                    )
        else:
            with open(os.path.join(exp_dir, f'{d}.csv'), 'w') as fp:
                for setn in ['dev', 'eval']:
                    mt_hm = []
                    fp.write('-' * 80 + '\n')
                    fp.write(f'{setn.upper()}\n\n')
                    for mt in SET_MT_MAP['dcase21'][setn]:
                        info1 = {'sec': unisec[d][setn][mt].tolist() + ['hmean']}
                        info1.update({keys[i]: result[d][setn][mt][:, i].tolist() for i in range(5)})
                        df1 = pd.DataFrame(info1)
                        mt_hm.append(result[d][setn][mt][-1, -1])
                        fp.write(f'{setn}_{mt}: {result[d][setn][mt][-1, -1]:.2f}\n')
                        fp.write(df1.to_csv(index=False))
                        fp.write('\n')
                    subset_hm[setn] = float(hmean(mt_hm))
                    fp.write(f'subset_hm: {subset_hm[setn]:.2f}\n\n')
                    if score_vis:
                        vis_score(
                            scores=scores[d][setn],
                            result=result[d][setn],
                            setn=setn,
                            out_dir=exp_dir
                        )
                fp.write('-' * 80 + '\n')
                fp.write(f"total_hm: {hmean([subset_hm['dev'], subset_hm['eval']]):.2f}\n")
        total_hm[d] = copy.deepcopy(subset_hm)
    return result, total_hm


def cal_aucs_dcase22(
    score_dict: Dict[str, Dict[str, pd.DataFrame]],
    exp_dir: Optional[str],
    score_vis: bool = True
):
    # score_dict: {detector: {setn_mt_sett: DataFrame}}
    result, result_dom = {}, {}
    unisec, scores = {}, {}
    for d in score_dict.keys():  # detector
        result[d], result_dom[d] = {}, {}
        scores[d], unisec[d] = {}, {}
        for setn in ['dev', 'eval']:
            result[d][setn], result_dom[d][setn] = {}, {}
            unisec[d][setn], scores[d][setn] = {}, {}
            for mt in SET_MT_MAP['dcase22'][setn]:
                result[d][setn][mt], result_dom[d][setn][mt] = [], []
                scores[d][setn][mt] = {}
                df = score_dict[d][f'{setn}_{mt}_test']
                unisec[d][setn][mt] = np.unique(df['sec'])
                for sec in unisec[d][setn][mt]:
                    df_sec = df.loc[df['sec'] == sec]
                    # DCASE22 benchmark
                    df_sou = df_sec.loc[(df['domain'] == 0) | (df['status'] == 1)]
                    df_tar = df_sec.loc[(df['domain'] == 1) | (df['status'] == 1)]
                    auc_s = roc_auc_score(df_sou['status'], df_sou['score']) * 100
                    auc_t = roc_auc_score(df_tar['status'], df_tar['score']) * 100
                    pauc = roc_auc_score(df_sec['status'], df_sec['score'], max_fpr=0.1) * 100
                    hm = hmean([auc_s, auc_t, pauc])
                    result[d][setn][mt].append([auc_s, auc_t, pauc, hm])
                    scores[d][setn][mt][sec] = parse_scores(df_sou['status'], df_sou['score'])
                    scores[d][setn][mt][sec].extend(parse_scores(df_tar['status'], df_tar['score']))
                    # Domain-wise benchmark
                    df_dom_sou = df_sec.loc[df['domain'] == 0]
                    df_dom_tar = df_sec.loc[df['domain'] == 1]
                    auc_dom_s = roc_auc_score(df_dom_sou['status'], df_dom_sou['score']) * 100
                    auc_dom_t = roc_auc_score(df_dom_tar['status'], df_dom_tar['score']) * 100
                    result_dom[d][setn][mt].append([auc_dom_s, auc_dom_t])
                mt_result = np.array(result[d][setn][mt])
                result[d][setn][mt].append([hmean(mt_result[:, i]) for i in range(4)])
                result[d][setn][mt] = np.array(result[d][setn][mt]).round(2)

                mt_result_dom = np.array(result_dom[d][setn][mt])
                result_dom[d][setn][mt].append([hmean(mt_result_dom[:, i]) for i in range(2)])
                result_dom[d][setn][mt] = np.array(result_dom[d][setn][mt]).round(2)

    keys = ['auc_s', 'auc_t', 'pauc', 'hmean']
    total_hm = {}
    for d in score_dict.keys():
        subset_hm = {}
        if exp_dir is None:
            for setn in ['dev', 'eval']:
                subset_hm[setn] = hmean([
                    result[d][setn][mt][-1, -1]
                    for mt in SET_MT_MAP['dcase22'][setn]
                ])
                if score_vis:
                    vis_score(
                        scores=scores[d][setn],
                        result=result[d][setn],
                        setn=setn,
                        out_dir=exp_dir
                    )
        else:
            with open(os.path.join(exp_dir, f'{d}.csv'), 'w') as fp:
                for setn in ['dev', 'eval']:
                    mt_hm = []
                    fp.write('-' * 80 + '\n')
                    fp.write(f'{setn.upper()}\n\n')
                    for mt in SET_MT_MAP['dcase22'][setn]:
                        info1 = {'sec': unisec[d][setn][mt].tolist() + ['hmean']}
                        info1.update({keys[i]: result[d][setn][mt][:, i].tolist() for i in range(4)})
                        df1 = pd.DataFrame(info1)
                        mt_hm.append(result[d][setn][mt][-1, -1])
                        fp.write(f'DCASE22 benchmark {setn}_{mt}: {result[d][setn][mt][-1, -1]:.2f}\n')
                        fp.write(df1.__repr__())
                        fp.write('\n')
                        fp.write('Domain-wise benchmark\n')
                        info2 = {'sec': unisec[d][setn][mt].tolist() + ['hmean']}
                        info2.update({keys[i]: result_dom[d][setn][mt][:, i].tolist() for i in range(2)})
                        df2 = pd.DataFrame(info2)
                        fp.write(df2.__repr__())
                        fp.write('\n' + '-' * 40 + '\n')
                    subset_hm[setn] = float(hmean(mt_hm))
                    fp.write(f'subset_hm: {subset_hm[setn]:.2f}\n\n')
                    if score_vis:
                        vis_score(
                            scores=scores[d][setn],
                            result=result[d][setn],
                            setn=setn,
                            out_dir=exp_dir
                        )
                fp.write('-' * 80 + '\n')
                fp.write(f"total_hm: {hmean([subset_hm['dev'], subset_hm['eval']]):.2f}\n")
        total_hm[d] = copy.deepcopy(subset_hm)
    return result, total_hm


def cal_aucs_dcase232425(
    score_dict: Dict[str, Dict[str, pd.DataFrame]],
    exp_dir: Optional[str],
    dataset: str,
    score_vis: bool = True,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, float]]]:
    '''Calculate the scores following the challenge rules of DCASE 2023, 2024

    Args:
        score_dict (Dict[str, Dict[str, pd.DataFrame]]): {detector: {setn_mt_sett: meta data + score}}
        exp_dir (Optional[str]): directory to write detection results
        dataset (str): which dataset
        score_vis (bool, optional): plot score distribution. Defaults to True.

    Returns:
        result: {detector: {setn: np.ndarray}}, [auc_s, auc_t, pauc, hmean]
        total_hm: {detector: {setn: setn_hmean}}
    '''
    assert dataset in ['dcase23', 'dcase24', 'dcase25']
    result, result_dom = {}, {}
    unisec, scores = {}, {}
    for d in score_dict.keys():
        result[d], result_dom[d] = {}, {}
        unisec[d], scores[d] = {}, {}
        for setn in ['dev', 'eval']:
            result[d][setn], result_dom[d][setn] = [], []
            unisec[d][setn], scores[d][setn] = {}, {}
            for mt in SET_MT_MAP[dataset][setn]:
                scores[d][setn][mt] = {}
                df = score_dict[d][f'{setn}_{mt}_test']
                unisec[d][setn][mt] = np.unique(df['sec'])
                for sec in unisec[d][setn][mt]:
                    df_sec = df.loc[df['sec'] == sec]
                    # Challenge benchmark
                    df_sou = df_sec.loc[(df['domain'] == 0) | (df['status'] == 1)]
                    df_tar = df_sec.loc[(df['domain'] == 1) | (df['status'] == 1)]
                    auc_s = roc_auc_score(df_sou['status'], df_sou['score']) * 100
                    auc_t = roc_auc_score(df_tar['status'], df_tar['score']) * 100
                    pauc = roc_auc_score(df_sec['status'], df_sec['score'], max_fpr=0.1) * 100
                    hm = hmean([auc_s, auc_t, pauc])
                    result[d][setn].append([auc_s, auc_t, pauc, hm])
                    scores[d][setn][mt][sec] = parse_scores(df_sou['status'], df_sou['score'])
                    scores[d][setn][mt][sec].extend(parse_scores(df_tar['status'], df_tar['score']))
                    # Domain-wise benchmark
                    df_dom_sou = df_sec.loc[df['domain'] == 0]
                    df_dom_tar = df_sec.loc[df['domain'] == 1]
                    auc_dom_s = roc_auc_score(df_dom_sou['status'], df_dom_sou['score']) * 100
                    auc_dom_t = roc_auc_score(df_dom_tar['status'], df_dom_tar['score']) * 100
                    result_dom[d][setn].append([auc_dom_s, auc_dom_t])
            full_result = np.array(result[d][setn])
            result[d][setn].append([hmean(full_result[:, i]) for i in range(4)])
            result[d][setn] = np.array(result[d][setn]).round(2)

            full_result_dom = np.array(result_dom[d][setn])
            result_dom[d][setn].append([hmean(full_result_dom[:, i]) for i in range(2)])
            result_dom[d][setn] = np.array(result_dom[d][setn]).round(2)

    keys = ['auc_s', 'auc_t', 'pauc', 'hmean']
    total_hm = {}
    for d in score_dict.keys():
        if exp_dir is not None:
            with open(os.path.join(exp_dir, f'{d}.csv'), 'w') as fp:
                for setn in ['dev', 'eval']:
                    fp.write('-' * 80 + '\n')
                    fp.write(f'{setn.upper()}\n\n')
                    info1 = {'mt': SET_MT_MAP[dataset][setn] + ['hmean']}
                    info1.update({keys[i]: result[d][setn][:, i].tolist() for i in range(4)})
                    df1 = pd.DataFrame(info1)
                    fp.write(f'{dataset.upper()} benchmark\n')
                    fp.write(df1.__repr__())
                    fp.write('\n\n')
                    fp.write('Domain-wise benchmark\n')
                    info2 = {'mt': SET_MT_MAP[dataset][setn] + ['hmean']}
                    info2.update({keys[i]: result_dom[d][setn][:, i].tolist() for i in range(2)})
                    df2 = pd.DataFrame(info2)
                    fp.write(df2.__repr__())
                    fp.write('\n\n')
                    if score_vis:
                        mt_result = {
                            mt: result[d][setn][i][np.newaxis, :]
                            for i, mt in enumerate(SET_MT_MAP[dataset][setn])
                        }
                        vis_score(scores[d][setn], mt_result, setn, exp_dir)
                fp.write('-' * 80 + '\n')
                fp.write(f"total_hm: {hmean([result[d]['dev'][-1, -1], result[d]['eval'][-1, -1]]):.2f}\n")
        total_hm[d] = {
            'dev': float(result[d]['dev'][-1, -1]),
            'eval': float(result[d]['eval'][-1, -1])
        }
    return result, total_hm


def cal_aucs_dcase25(
    score_dict: Dict[str, Dict[str, pd.DataFrame]],
    exp_dir: Optional[str],
    score_vis: bool = True
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, float]]]:
    '''Calculate the scores following the challenge rule.

    Args:
        score_dict (Dict[str, Dict[str, pd.DataFrame]]): {detector: {setn_mt_sett: meta data + score}}
        exp_dir (Optional[str]): directory to write detection results
        score_vis (bool, optional): plot score distribution. Defaults to True.

    Returns:
        result: {detector: {setn: np.ndarray}}, [auc_s, auc_t, pauc, hmean]
        total_hm: {detector: {setn: setn_hmean}}
    '''
    result, result_dom = {}, {}
    unisec, scores = {}, {}
    for d in score_dict.keys():
        result[d], result_dom[d] = {}, {}
        unisec[d], scores[d] = {}, {}
        for setn in ['dev']:  # NOTE:
            result[d][setn], result_dom[d][setn] = [], []
            unisec[d][setn], scores[d][setn] = {}, {}
            for mt in SET_MT_MAP['dcase25'][setn]:
                scores[d][setn][mt] = {}
                df = score_dict[d][f'{setn}_{mt}_test']
                unisec[d][setn][mt] = np.unique(df['sec'])
                for sec in unisec[d][setn][mt]:
                    df_sec = df.loc[df['sec'] == sec]
                    # Challenge benchmark
                    df_sou = df_sec.loc[(df['domain'] == 0) | (df['status'] == 1)]
                    df_tar = df_sec.loc[(df['domain'] == 1) | (df['status'] == 1)]
                    auc_s = roc_auc_score(df_sou['status'], df_sou['score']) * 100
                    auc_t = roc_auc_score(df_tar['status'], df_tar['score']) * 100
                    pauc = roc_auc_score(df_sec['status'], df_sec['score'], max_fpr=0.1) * 100
                    hm = hmean([auc_s, auc_t, pauc])
                    result[d][setn].append([auc_s, auc_t, pauc, hm])
                    scores[d][setn][mt][sec] = parse_scores(df_sou['status'], df_sou['score'])
                    scores[d][setn][mt][sec].extend(parse_scores(df_tar['status'], df_tar['score']))
                    # Domain-wise benchmark
                    df_dom_sou = df_sec.loc[df['domain'] == 0]
                    df_dom_tar = df_sec.loc[df['domain'] == 1]
                    auc_dom_s = roc_auc_score(df_dom_sou['status'], df_dom_sou['score']) * 100
                    auc_dom_t = roc_auc_score(df_dom_tar['status'], df_dom_tar['score']) * 100
                    result_dom[d][setn].append([auc_dom_s, auc_dom_t])
            full_result = np.array(result[d][setn])
            result[d][setn].append([hmean(full_result[:, i]) for i in range(4)])
            result[d][setn] = np.array(result[d][setn]).round(2)

            full_result_dom = np.array(result_dom[d][setn])
            result_dom[d][setn].append([hmean(full_result_dom[:, i]) for i in range(2)])
            result_dom[d][setn] = np.array(result_dom[d][setn]).round(2)

    keys = ['auc_s', 'auc_t', 'pauc', 'hmean']
    total_hm = {}  # {detector: {dev: xxx, eval: yyy}}
    for d in score_dict.keys():
        if exp_dir is not None:
            with open(os.path.join(exp_dir, f'{d}.csv'), 'w') as fp:
                for setn in ['dev']:  # NOTE:
                    fp.write('-' * 80 + '\n')
                    fp.write(f'{setn.upper()}\n\n')
                    info1 = {'mt': SET_MT_MAP['dcase25'][setn] + ['hmean']}
                    info1.update({keys[i]: result[d][setn][:, i].tolist() for i in range(4)})
                    df1 = pd.DataFrame(info1)
                    fp.write('DCASE25 benchmark\n')
                    fp.write(df1.__repr__())
                    fp.write('\n\n')
                    fp.write('Domain-wise benchmark\n')
                    info2 = {'mt': SET_MT_MAP['dcase25'][setn] + ['hmean']}
                    info2.update({keys[i]: result_dom[d][setn][:, i].tolist() for i in range(2)})
                    df2 = pd.DataFrame(info2)
                    fp.write(df2.__repr__())
                    fp.write('\n\n')
                    if score_vis:
                        mt_result = {
                            mt: result[d][setn][i][np.newaxis, :]
                            for i, mt in enumerate(SET_MT_MAP['dcase25'][setn])
                        }
                        vis_score(scores[d][setn], mt_result, setn, exp_dir)
                fp.write('-' * 80 + '\n')
                # fp.write(f"total_hm: {hmean([result[d]['dev'][-1, -1], result[d]['eval'][-1, -1]]):.2f}\n")
        total_hm[d] = {
            'dev': float(result[d]['dev'][-1, -1]),
        }
    return result, total_hm
