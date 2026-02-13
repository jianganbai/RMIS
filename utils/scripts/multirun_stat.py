import os
import re
import glob
import pickle
import argparse
import numpy as np
import pandas as pd

from scipy.stats import hmean
from typing import Iterable, Union, List, Optional

from uni_detect.task.dcase.common import SET_MT_MAP


def num_mean(data: Iterable) -> Union[float, str]:
    '''Arithmetic mean for iterable with string elements

    Args:
        data (Iterable): incoming data. containing int, float, str

    Returns:
        Union[float, str]: arithmetic mean for only numeric data, or '-'
    '''
    num_data = [d for d in data if isinstance(d, (int, float))]
    if len(num_data) > 0:
        return float(np.mean(num_data))
    else:
        return '-'


def num_hmean(data: Iterable) -> Union[float, str]:
    '''Hmean for iterable with string elements

    Args:
        data (Iterable): incoming data. containing int, float, str

    Returns:
        Union[float, str]: hmean for only numeric data, or '-'
    '''
    num_data = [d for d in data if isinstance(d, (int, float))]
    if len(num_data) > 0:
        return hmean(num_data)
    else:
        return '-'


def num_stdev(data: Iterable) -> Union[float, str]:
    '''Standard deviation for iterable with string elements

    Args:
        data (Iterable): incoming data. containing int, float, str

    Returns:
        Union[float, str]: std for only numeric data, or '-'
    '''
    num_data = [d for d in data if isinstance(d, (int, float))]
    if len(num_data) > 1:
        return float(np.std(num_data, ddof=1))
    else:
        return '-'


def stat_dcase202122(
    step_list: List[str],
    ds: str,
    exp_dir: str,
    save_note: Optional[str]
) -> pd.DataFrame:
    MT_MAP = SET_MT_MAP[ds]
    rf_list = [os.path.join(s, ds, 'result.pkl') for s in step_list]
    rr_list = []

    for rf in rf_list:
        with open(rf, 'rb') as fp:
            r = pickle.load(fp)
        rr = {}
        for setn in ['dev', 'eval']:
            if setn in r.keys():
                assert set(MT_MAP[setn]) == set(r[setn].keys()), \
                    f'Machine types {set(r[setn].keys())} is incomplete for {ds}'
                mta = []
                for mt, a in r[setn].items():
                    rr[(setn, mt)] = a[-1, -1]
                    mta.append(a[-1, -1])
                if ds == 'dcase20':
                    rr[(setn, 'subset')] = num_mean(mta)  # DCASE20 uses arithmetic mean
                else:
                    rr[(setn, 'subset')] = num_hmean(mta)
            else:
                for mt in MT_MAP[setn]:
                    rr[(setn, mt)] = '-'
                rr[(setn, 'subset')] = '-'
        if ds == 'dcase20':
            rr[('all', 'mean')] = num_mean([rr[(setn, 'subset')] for setn in ['dev', 'eval']])
        else:
            rr[('all', 'hmean')] = num_hmean([rr[(setn, 'subset')] for setn in ['dev', 'eval']])
        rr_list.append(pd.Series(rr))

    run_df = pd.DataFrame(rr_list)
    mean_row = run_df.apply(num_mean, axis=0)
    std_row = run_df.apply(num_stdev, axis=0)
    df = pd.DataFrame(rr_list + [mean_row, std_row])
    df = df.map(lambda x: round(x, 2))
    df['index'] = [f'run{i}' for i in range(len(rf_list))] + ['mean', 'std']
    df.set_index('index', inplace=True)
    df = df.T

    if save_note is None:
        save_path = os.path.join(exp_dir, f'{ds}_stat.csv')
    else:
        save_path = os.path.join(exp_dir, f'{ds}_{save_note}_stat.csv')
    with open(save_path, 'w') as fp:
        fp.write(df.to_string())
    return df


def stat_dcase232425(
    step_list: List[str],
    ds: str,
    exp_dir: str,
    save_note: Optional[str]
) -> pd.DataFrame:
    MT_MAP = SET_MT_MAP[ds]
    rf_list = [os.path.join(s, ds, 'result.pkl') for s in step_list]
    rr_list = []

    for rf in rf_list:
        with open(rf, 'rb') as fp:
            r = pickle.load(fp)
        rr = {}
        for setn in ['dev', 'eval']:
            if setn in r.keys():
                assert len(MT_MAP[setn]) == (r[setn].shape[0] - 1)
                for i, mt in enumerate(MT_MAP[setn]):
                    rr[(setn, mt)] = r[setn][i, -1]
                rr[(setn, 'subset')] = r[setn][-1, -1]
            else:
                for mt in MT_MAP[setn]:
                    rr[(setn, mt)] = '-'
                rr[(setn, 'subset')] = '-'
        rr[('all', 'hmean')] = num_hmean([rr[(setn, 'subset')] for setn in ['dev', 'eval']])
        rr_list.append(pd.Series(rr))

    run_df = pd.DataFrame(rr_list)
    mean_row = run_df.apply(num_mean, axis=0)
    std_row = run_df.apply(num_stdev, axis=0)
    df = pd.DataFrame(rr_list + [mean_row, std_row])
    df = df.map(lambda x: round(x, 2))
    df['index'] = [f'run{i}' for i in range(len(rf_list))] + ['mean', 'std']
    df.set_index('index', inplace=True)
    df = df.T

    if save_note is None:
        save_path = os.path.join(exp_dir, f'{ds}_stat.csv')
    else:
        save_path = os.path.join(exp_dir, f'{ds}_{save_note}_stat.csv')
    with open(save_path, 'w') as fp:
        fp.write(df.to_string())
    return df


def stat_dcase25(
    step_list: List[str],
    ds: str,
    exp_dir: str,
    save_note: Optional[str]
) -> pd.DataFrame:
    raise Exception('deprecated!')
    # TODO: merge with stat_dcase2324 after challenge
    MT_MAP = SET_MT_MAP[ds]
    rf_list = [os.path.join(s, ds, 'result.pkl') for s in step_list]
    rr_list = []

    for rf in rf_list:
        with open(rf, 'rb') as fp:
            r = pickle.load(fp)
        rr = {}
        for setn in ['dev']:
            if setn in r.keys():
                assert len(MT_MAP[setn]) == (r[setn].shape[0] - 1)
                for i, mt in enumerate(MT_MAP[setn]):
                    rr[(setn, mt)] = r[setn][i, -1]
                rr[(setn, 'subset')] = r[setn][-1, -1]
            else:
                for mt in MT_MAP[setn]:
                    rr[(setn, mt)] = '-'
                rr[(setn, 'subset')] = '-'
        rr[('all', 'hmean')] = num_hmean([rr[(setn, 'subset')] for setn in ['dev']])
        rr_list.append(pd.Series(rr))

    run_df = pd.DataFrame(rr_list)
    mean_row = run_df.apply(num_mean, axis=0)
    std_row = run_df.apply(num_stdev, axis=0)
    df = pd.DataFrame(rr_list + [mean_row, std_row])
    df = df.map(lambda x: round(x, 2))
    df['index'] = [f'run{i}' for i in range(len(rf_list))] + ['mean', 'std']
    df.set_index('index', inplace=True)
    df = df.T

    if save_note is None:
        save_path = os.path.join(exp_dir, f'{ds}_stat.csv')
    else:
        save_path = os.path.join(exp_dir, f'{ds}_{save_note}_stat.csv')
    with open(save_path, 'w') as fp:
        fp.write(df.to_string())
    return df


def stat_dcase(step_list: List[str], exp_dir: str, save_note: Optional[str]) -> None:
    demo_rf_list = glob.glob(os.path.join(step_list[0], 'dcase[0-9][0-9]/result.pkl'))
    ds_list = sorted(
        [p.split('/')[-2] for p in demo_rf_list],
        key=lambda x: int(x[-2:])
    )
    rr_list = []
    for ds in ds_list:
        if ds in ['dcase20', 'dcase21', 'dcase22']:
            dsdf = stat_dcase202122(step_list, ds, exp_dir, save_note)
        elif ds in ['dcase23', 'dcase24', 'dcase25']:
            dsdf = stat_dcase232425(step_list, ds, exp_dir, save_note)
        elif ds in ['dcase25']:
            dsdf = stat_dcase25(step_list, ds, exp_dir, save_note)
        rr = dsdf.iloc[len(dsdf) - 1]
        rr_list.append(rr)
    df = pd.DataFrame(rr_list)
    df.loc[len(df)] = df.apply(num_mean, axis=0)
    df.iloc[len(df) - 1, len(df.columns) - 1] = num_stdev(
        df.iloc[len(df) - 1, 0: len(df.columns) - 2].to_list()
    )
    df = df.map(lambda x: round(x, 2))
    df.index = ds_list + ['mean']
    if save_note is None:
        save_path = os.path.join(exp_dir, 'dcase_stat.csv')
    else:
        save_path = os.path.join(exp_dir, f'dcase_{save_note}_stat.csv')
    with open(save_path, 'w') as fp:
        fp.write(df.to_string())


def decode_str_df(rf_file: str) -> pd.Series:
    with open(rf_file, 'r') as fp:
        rf_str = fp.read()
    df_str = rf_str[:re.search('-' * 10, rf_str).start()]
    acc_dict = {}
    for row in df_str.strip().split('\n')[1:]:
        elems = row.split()
        acc_dict[elems[1]] = float(elems[-1])
    acc_dict['macro-acc'] = float(re.search(r'macro-acc: ([\d.]*)', rf_str).group(1))
    acc_sri = pd.Series(acc_dict)
    return acc_sri


def stat_cls(
    step_list: List[str],
    exp_dir: str,
    prefix: str,
    read_key: str = 'acc',
    out_key: str = 'stat',
) -> None:
    rf_list = [os.path.join(s, f'{read_key}.csv') for s in step_list]
    rr_list = []
    for rf in rf_list:
        rr_list.append(decode_str_df(rf))
    run_df = pd.DataFrame(rr_list)
    mean_row = run_df.apply(num_mean, axis=0)
    std_row = run_df.apply(num_stdev, axis=0)
    df = pd.DataFrame(rr_list + [mean_row, std_row])
    df = df.map(lambda x: round(x, 2) if isinstance(x, float) else x)
    df['index'] = [f'{prefix}{i}' for i in range(len(rf_list))] + ['mean', 'std']
    df.set_index('index', inplace=True)
    df = df.T

    with open(os.path.join(exp_dir, f'{prefix}_{out_key}.csv'), 'w') as fp:
        fp.write(df.to_string())


def main(
    ds: Optional[str],
    exp_dir: str,
    prefix: str,
    result_dir: Optional[str],
):
    assert os.path.exists(exp_dir)
    runs = glob.glob(os.path.join(exp_dir, f'{prefix}*/'))
    runs = [r for r in runs if re.match(rf'{prefix}\d', os.path.basename(r[:-1]))]
    runs = sorted(runs, key=lambda x: int(os.path.basename(x[:-1])[len(prefix):]))
    step_list = []
    if result_dir is not None:
        for run_dir in runs:
            step_dir = os.path.join(run_dir, result_dir)
            if os.path.exists(step_dir):
                step_list.append(step_dir)
    else:
        for run_dir in runs:
            ckpt_list = sorted(
                glob.glob(os.path.join(run_dir, 'saved_models/step_*')),
                key=lambda x: float(os.path.splitext(os.path.basename(x))[0].split('_')[-1])
            )
            step = f"step_{os.path.basename(ckpt_list[-1]).split('_')[1]}"
            step_list.append(os.path.join(run_dir, step))

    if ds in ['dcase20', 'dcase21', 'dcase22']:
        stat_dcase202122(step_list, ds, exp_dir, result_dir)
    elif ds in ['dcase23', 'dcase24', 'dcase25']:
        stat_dcase232425(step_list, ds, exp_dir, result_dir)
    elif ds in ['dcase25']:
        stat_dcase25(step_list, ds, exp_dir, result_dir)
    elif ds == 'dcase':
        stat_dcase(step_list, exp_dir, result_dir)
    elif ds == 'gwbw':
        stat_cls(step_list, exp_dir, prefix=prefix, read_key='test1', out_key='stat_test1')
        stat_cls(step_list, exp_dir, prefix=prefix, read_key='test2', out_key='stat_test2')
    else:
        stat_cls(step_list, exp_dir, prefix)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds', type=str, required=True)
    parser.add_argument('--exp_dir', type=str, required=True)
    parser.add_argument('--prefix', type=str, default='run')
    parser.add_argument('--result_dir', type=str, default=None)
    args = parser.parse_args()
    args = vars(args)

    main(**args)
