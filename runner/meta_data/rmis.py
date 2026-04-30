import os
import re
import time
import yaml
import glob
import queue
import struct
import argparse
import torchaudio
import pandas as pd
import multiprocessing as mp

from typing import Dict, Any
from collections import defaultdict


from uni_detect.task.dcase.common import (
    SET_MT_MAP, DCASE20_MT, DCASE21_MT,
    DCASE22_MT, DCASE23_MT, DCASE24_MT,
    DCASE25_MT
)


def parse_dcase_eval_gt(
    eval_gt_dir: str,
    dataset: str
) -> Dict[str, Dict[str, str]]:
    ws = dataset[-2:]
    num_element = {  # [num elem for file row, num elem for mt row]
        'dcase20': [3, 1], 'dcase21': [2, 1],
        'dcase22': [2, 1], 'dcase23': [2, 1],
        'dcase24': [2, 1], 'dcase25': [2, 1],
    }
    with open(os.path.join(eval_gt_dir, f'eval_data_list_20{ws}.csv'), 'r') as fp:
        gt = fp.readlines()

    gt = [g.rstrip().split(',') for g in gt]
    this_mt, eval_gt = None, {}
    for line in gt:
        if len(line) == num_element[dataset][0]:  # file
            eval_gt[this_mt][line[0]] = line[1]
        elif len(line) == num_element[dataset][1] and line[0] in SET_MT_MAP[dataset]['eval']:  # mt
            this_mt = line[0]
            eval_gt[this_mt] = {}
        else:
            raise Exception(f'Unknown line {line}')

    return eval_gt


def read_wav_info(wav: str) -> Dict[str, Any]:
    if wav.endswith('wav'):
        tolB = os.path.getsize(wav)
        with open(wav, 'rb') as fr:
            fr.seek(22, 0)
            nch = struct.unpack('<H', fr.read(2))[0]
            sr = struct.unpack('<L', fr.read(4))[0]
            fr.seek(34, 0)
            wd = (struct.unpack('<H', fr.read(2))[0] + 7) // 8
            tol_frames = (tolB - 44) // (nch * wd)
            audio_info = {'num_frames': tol_frames, 'sample_rate': sr}
    else:
        audio_info = torchaudio.info(wav)
        audio_info = vars(audio_info)
    audio_info['dur'] = audio_info['num_frames'] / audio_info['sample_rate']
    return audio_info


def wav_reader(path_queue, info_dict):
    while True:
        path = path_queue.get()
        if path is None:
            path_queue.put(None)
            break
        else:
            info_dict[path] = read_wav_info(path)


def gen_20(
    root_dir: str,
    eval_gt_dir: str,
    out_dir: str,
    read_dur: bool = True,
    num_readers: int = 4
) -> None:
    eval_gt = parse_dcase_eval_gt(eval_gt_dir, 'dcase20')
    if read_dur:
        manager = mp.Manager()
        path_queue = mp.Queue()
        info_dict = manager.dict()
        reader_list = [
            mp.Process(target=wav_reader, args=(path_queue, info_dict), daemon=True)
            for _ in range(num_readers)
        ]
        [r.start() for r in reader_list]

    for setn in ['dev', 'eval']:
        for sett in ['train', 'test']:
            os.makedirs(os.path.join(out_dir, setn, sett), exist_ok=True)
            for mt in DCASE20_MT[setn]:
                mt_dir = os.path.join(os.path.join(root_dir, f'{setn}_data'), mt, sett)
                data = defaultdict(list)
                wav_list = sorted([f for f in os.listdir(mt_dir) if f.endswith('.wav')])
                for wav in wav_list:
                    data['file'].append(os.path.join(f'{setn}_data/{mt}/{sett}/{wav}'))
                    data['path'].append(os.path.join(mt_dir, wav))
                    data['mt'].append(mt)
                    if setn == 'eval' and sett == 'test':
                        assert wav in eval_gt[mt].keys(), f'{wav} not in eval_gt!'
                        info = os.path.splitext(eval_gt[mt][wav])[0].split('_')
                    else:
                        info = os.path.splitext(wav)[0].split('_')
                    data['sec'].append(int(info[2]))
                    data['domain'].append(0)
                    data['status'].append(1 if info[0] == 'anomaly' else 0)
                    data['attri'].append('noAttribute_dcase20')
                    if read_dur:
                        path_queue.put(os.path.join(mt_dir, wav))

                if read_dur:
                    for path in data['path']:
                        if path not in info_dict:
                            time.sleep(0.5)
                        for k in info_dict[path].keys():
                            data[k].append(info_dict[path][k])
                    while not path_queue.empty():
                        try:
                            path_queue.get_nowait()
                        except queue.Empty:
                            break
                df = pd.DataFrame(data)
                out_file = os.path.join(out_dir, setn, sett, f'{mt}.csv')
                df.to_csv(out_file, index=False)

    if read_dur:
        path_queue.put(None)
        [r.join() for r in reader_list]


def gen_21(
    root_dir: str,
    eval_gt_dir: str,
    out_dir: str,
    read_dur: bool = True,
    num_readers: int = 4
) -> None:
    eval_gt = parse_dcase_eval_gt(eval_gt_dir, 'dcase21')
    if read_dur:
        manager = mp.Manager()
        path_queue = mp.Queue()
        info_dict = manager.dict()
        reader_list = [
            mp.Process(target=wav_reader, args=(path_queue, info_dict), daemon=True)
            for _ in range(num_readers)
        ]
        [r.start() for r in reader_list]

    for setn in ['dev', 'eval']:
        for sett in ['train', 'test']:
            os.makedirs(os.path.join(out_dir, setn, sett), exist_ok=True)
            for mt in DCASE21_MT[setn]:
                mt_dir = os.path.join(root_dir, f'{setn}_data', mt)
                data = defaultdict(list)
                if sett == 'train':
                    path_list = [
                        os.path.join(mt_dir, 'train', f)
                        for f in os.listdir(os.path.join(mt_dir, 'train')) if f.endswith('.wav')]
                else:
                    path_list = []
                    for d in ['source_test', 'target_test']:
                        path_list.extend([
                            os.path.join(mt_dir, d, f)
                            for f in os.listdir(os.path.join(mt_dir, d)) if f.endswith('.wav')
                        ])
                path_list = sorted(path_list)
                for path in path_list:
                    wav = os.path.basename(path)
                    data['file'].append(os.path.join('/'.join(path.split('/')[-4:])))
                    data['path'].append(path)
                    data['mt'].append(mt)
                    if setn == 'eval' and sett == 'test':
                        assert wav in eval_gt[mt].keys(), f'{wav} not in eval_gt!'
                        info = os.path.splitext(eval_gt[mt][wav])[0].split('_')
                    else:
                        info = os.path.splitext(wav)[0].split('_')
                    data['sec'].append(int(info[1]))
                    data['domain'].append(1 if info[2] == 'target' else 0)
                    data['status'].append(1 if info[4] == 'anomaly' else 0)
                    if len(info) > 6:
                        data['attri'].append('_'.join(info[6:])+'_dcase21')
                    else:
                        data['attri'].append('noAttribute_dcase21')
                    if read_dur:
                        path_queue.put(path)

                if read_dur:
                    for path in data['path']:
                        if path not in info_dict:
                            time.sleep(0.5)
                        for k in info_dict[path].keys():
                            data[k].append(info_dict[path][k])
                    while not path_queue.empty():
                        try:
                            path_queue.get_nowait()
                        except queue.Empty:
                            break
                df = pd.DataFrame(data)
                out_file = os.path.join(out_dir, setn, sett, f'{mt}.csv')
                df.to_csv(out_file, index=False)

    if read_dur:
        path_queue.put(None)
        [r.join() for r in reader_list]


def gen_222324(
    root_dir: str,
    eval_gt_dir: str,
    out_dir: str,
    read_dur: bool = False,
    num_readers: int = 4
) -> None:
    ds = os.path.basename(out_dir)
    SMT = {'dcase22': DCASE22_MT, 'dcase23': DCASE23_MT, 'dcase24': DCASE24_MT}
    eval_gt = parse_dcase_eval_gt(eval_gt_dir, ds)
    if read_dur:
        manager = mp.Manager()
        path_queue = mp.Queue()
        info_dict = manager.dict()
        reader_list = [
            mp.Process(target=wav_reader, args=(path_queue, info_dict), daemon=True)
            for _ in range(num_readers)
        ]
        [r.start() for r in reader_list]

    for setn in ['dev', 'eval']:
        for sett in ['train', 'test']:
            os.makedirs(os.path.join(out_dir, setn, sett), exist_ok=True)
            for mt in SMT[ds][setn]:
                mt_dir = os.path.join(os.path.join(root_dir, f'{setn}_data'), mt, sett)
                data = defaultdict(list)
                wav_list = sorted([f for f in os.listdir(mt_dir) if f.endswith('.wav')])
                for wav in wav_list:
                    data['file'].append(os.path.join(f'{setn}_data/{mt}/{sett}/{wav}'))
                    data['path'].append(os.path.join(mt_dir, wav))
                    data['mt'].append(mt)
                    if setn == 'eval' and sett == 'test':
                        assert wav in eval_gt[mt].keys(), f'{wav} not in eval_gt!'
                        info = os.path.splitext(eval_gt[mt][wav])[0].split('_')
                    else:
                        info = os.path.splitext(wav)[0].split('_')
                    data['sec'].append(int(info[1]))
                    data['domain'].append(1 if info[2] == 'target' else 0)
                    data['status'].append(1 if info[4] == 'anomaly' else 0)
                    data['attri'].append('_'.join(info[6:]) + f'_{ds}')
                    if read_dur:
                        path_queue.put(os.path.join(mt_dir, wav))

                if read_dur:
                    for path in data['path']:
                        if path not in info_dict:
                            time.sleep(0.5)
                        for k in info_dict[path].keys():
                            data[k].append(info_dict[path][k])
                    while not path_queue.empty():
                        try:
                            path_queue.get_nowait()
                        except queue.Empty:
                            break
                    info_dict.clear()
                df = pd.DataFrame(data)
                out_file = os.path.join(out_dir, setn, sett, f'{mt}.csv')
                df.to_csv(out_file, index=False)

    if read_dur:
        path_queue.put(None)
        [r.join() for r in reader_list]


def gen_25(
    root_dir: str,
    eval_gt_dir: str,
    out_dir: str,
    read_dur: bool = False,
    num_readers: int = 4
) -> None:
    eval_gt = parse_dcase_eval_gt(eval_gt_dir, 'dcase25')
    if read_dur:
        manager = mp.Manager()
        path_queue = mp.Queue()
        info_dict = manager.dict()
        reader_list = [
            mp.Process(target=wav_reader, args=(path_queue, info_dict), daemon=True)
            for _ in range(num_readers)
        ]
        [r.start() for r in reader_list]

    for setn in ['dev', 'eval']:
        for mt in DCASE25_MT[setn]:
            for sett in ['train', 'test']:
                os.makedirs(os.path.join(out_dir, setn, sett), exist_ok=True)
                data = defaultdict(list)
                for path in sorted(glob.glob(
                    os.path.join(root_dir, f'{setn}_data', mt, sett, '*.wav')
                )):
                    data['file'].append(os.path.relpath(path, root_dir))
                    data['path'].append(path)
                    data['mt'].append(mt)
                    wav = os.path.basename(path)
                    if setn == 'eval' and sett == 'test':
                        assert wav in eval_gt[mt].keys(), f'{wav} not in eval_gt!'
                        info = os.path.splitext(eval_gt[mt][wav])[0].split('_')
                    else:
                        info = os.path.splitext(wav)[0].split('_')
                    data['sec'].append(int(info[1]))
                    data['domain'].append(1 if 'target' in info else 0)
                    data['status'].append(1 if 'anomaly' in info else 0)
                    if len(info) > 6:
                        data['attri'].append('_'.join(info[6:]) + '_dcase25')
                    else:
                        data['attri'].append('noAttribute_dcase25')
                    if read_dur:
                        path_queue.put(path)

                if read_dur:
                    for path in data['path']:
                        if path not in info_dict:
                            time.sleep(0.5)
                        for k in info_dict[path].keys():
                            data[k].append(info_dict[path][k])
                    while not path_queue.empty():
                        try:
                            path_queue.get_nowait()
                        except queue.Empty:
                            break
                df = pd.DataFrame(data)
                out_file = os.path.join(out_dir, setn, sett, f'{mt}.csv')
                df.to_csv(out_file, index=False)

            # supplementary data
            os.makedirs(os.path.join(out_dir, setn, 'supple'), exist_ok=True)
            supple_data = []
            p = re.compile(r'section_\d+_(\D*)_\d+(.*).wav')
            for path in sorted(glob.glob(
                os.path.join(root_dir, f'{setn}_data', mt, 'supplemental/*.wav')
            )):
                file = os.path.relpath(path, root_dir)
                info = os.path.splitext(os.path.basename(path))[0].split('_')
                m = p.match(os.path.basename(file))
                assert m is not None
                file_type, attri = m.group(1), m.group(2).lstrip('_')
                attri += '_sup_dcase25'
                if read_dur:
                    path_queue.put(path)
                supple_data.append({
                    'file': file, 'type': file_type, 'mt': mt, 'sec': 0, 'attri': attri,
                    'path': path
                })

            if read_dur:
                for i in range(len(supple_data)):
                    path = supple_data[i]['path']
                    if path not in info_dict:
                        time.sleep(0.5)
                    for k in info_dict[path].keys():
                        supple_data[i][k] = info_dict[path][k]
                while not path_queue.empty():
                    try:
                        path_queue.get_nowait()
                    except queue.Empty:
                        break
                info_dict.clear()
            supple_df = pd.DataFrame(supple_data)
            out_file = os.path.join(out_dir, setn, 'supple', f'{mt}.csv')
            supple_df.to_csv(out_file, index=False)

    if read_dur:
        path_queue.put(None)
        [r.join() for r in reader_list]


def gen_idmt_isa_compressed_air(
    root_dir: str,
    out_dir: str,
    read_dur: bool = False,
    num_readers: int = 4
) -> None:
    # https://www.idmt.fraunhofer.de/en/publications/datasets/isa-compressed-air.html
    os.makedirs(out_dir, exist_ok=True)
    if read_dur:
        manager = mp.Manager()
        path_queue = mp.Queue()
        info_dict = manager.dict()
        reader_list = [
            mp.Process(target=wav_reader, args=(path_queue, info_dict), daemon=True)
            for _ in range(num_readers)
        ]
        [r.start() for r in reader_list]

    info = defaultdict(list)
    for leak in sorted(os.listdir(root_dir)):
        for noise in sorted(os.listdir(os.path.join(root_dir, leak))):
            if os.path.isdir(os.path.join(root_dir, leak, noise)):
                for sess in sorted(os.listdir(os.path.join(root_dir, leak, noise))):
                    dirn = os.path.join(root_dir, leak, noise, sess)
                    for f in sorted(os.listdir(os.path.join(dirn))):
                        elems = f.split('_')
                        info['file'].append(os.path.join(leak, noise, sess, f))
                        info['leak'].append(leak)
                        info['noise'].append(noise)
                        info['session'].append(sess)
                        info['knob'].append(elems[2])
                        info['mic'].append(elems[3])
                        info['status'].append('leak' if elems[1] == 'niO' else 'noleak')
                        info['scene'].append(f"{info['leak'][-1]}_{info['status'][-1]}")
                        info['path'].append(os.path.join(dirn, f))
                        info['ori'].append(os.path.join(
                            leak, noise, sess,
                            '_'.join(f.split('_')[:-1]) + '.' + f.split('.')[-1]
                        ))
                        if read_dur:
                            path_queue.put(os.path.join(dirn, f))

    if read_dur:
        for path in info['path']:
            if path not in info_dict:
                time.sleep(0.5)
            for k in info_dict[path].keys():
                info[k].append(info_dict[path][k])
        while not path_queue.empty():
            try:
                path_queue.get_nowait()
            except queue.Empty:
                break
    df = pd.DataFrame(info)
    df.to_csv(os.path.join(out_dir, 'all.csv'), index=False)

    if read_dur:
        path_queue.put(None)
        [r.join() for r in reader_list]


def gen_idmt_isa_electric_engine(
    root_dir: str,
    out_dir: str,
    read_dur: bool = False
) -> None:
    # https://www.idmt.fraunhofer.de/en/publications/datasets/isa-electric-engine.html
    os.makedirs(out_dir, exist_ok=True)
    info = {sett: defaultdict(list) for sett in ['train', 'test']}
    for sett in sorted(os.listdir(root_dir)):
        if 'cut' not in sett:
            continue
        for status in sorted(os.listdir(os.path.join(root_dir, sett))):
            for f in sorted(os.listdir(os.path.join(root_dir, sett, status))):
                ws = sett.split('_')[0]
                info[ws]['file'].append(os.path.join(sett, status, f))
                info[ws]['scene'].append(status.split('_')[1])
                info[ws]['path'].append(os.path.join(root_dir, sett, status, f))
                if read_dur:
                    audio_info = read_wav_info(os.path.join(root_dir, sett, status, f))
                    info[ws]['dur'].append(audio_info['num_frames'] / audio_info['sample_rate'])
    for sett in ['train', 'test']:
        df = pd.DataFrame(info[sett])
        df.to_csv(os.path.join(out_dir, f'{sett}.csv'), index=False)


def gen_wt_plane_gearbox(
    root_dir: str,
    out_dir: str,
    read_dur: bool = False
) -> None:
    # https://github.com/Liudd-BJUT/WT-planetary-gearbox-dataset
    cls_list = ['broken', 'healthy', 'missing_tooth', 'root_crack', 'wear']

    os.makedirs(out_dir, exist_ok=True)
    path_list = sorted(glob.glob(os.path.join(root_dir, '*.flac')))
    path_list = [p for p in path_list if 'keyphase' not in os.path.basename(p)]  # remove keyphase data
    info = defaultdict(list)
    for path in path_list:
        file = os.path.basename(path)
        for cls in cls_list:
            if file.startswith(cls):
                scene = cls
                break
        status = 'normal' if scene == 'healthy' else 'anomaly'
        info['file'].append(file)
        info['scene'].append(scene)
        info['status'].append(status)
        info['path'].append(path)
        info['ori'].append('_'.join(file.split('_')[:-2]) + '.flac')
        if read_dur:
            audio_info = read_wav_info(path)
            info['dur'].append(round(audio_info['num_frames'] / audio_info['sample_rate'], 1))
    df = pd.DataFrame(info)
    df.to_csv(os.path.join(out_dir, 'all.csv'), index=False)


def gen_mafaulda(
    root_dir: str,
    out_dir: str,
    read_dur: bool = False
) -> None:
    # https://www02.smt.ufrj.br/~offshore/mfs/page_01.html#SEC2
    cls_list = [
        'normal', 'horizontal-misalignment', 'vertical-misalignment', 'imbalance',
        'overhang_ball_fault', 'overhang_cage_fault', 'overhang_outer_race',  # sub fault
        'underhang_ball_fault', 'underhang_cage_fault', 'underhang_outer_race'  # sub fault
    ]

    os.makedirs(out_dir, exist_ok=True)
    path_list = sorted(glob.glob(os.path.join(root_dir, '**', '*.wav'), recursive=True))
    info = defaultdict(list)
    for path in path_list:
        file = os.path.basename(path)
        for cls in cls_list:
            if file.startswith(cls):
                scene = cls
                break
        info['file'].append(file)
        info['scene'].append(scene)
        info['status'].append('normal' if scene == 'normal' else 'anomaly')
        info['path'].append(path)
        info['ori'].append(file)
        if read_dur:
            audio_info = read_wav_info(path)
            info['dur'].append(round(audio_info['num_frames'] / audio_info['sample_rate'], 1))
    df = pd.DataFrame(info)
    df.to_csv(os.path.join(out_dir, 'all.csv'), index=False)


def gen_sdust_bearing(
    root_dir: str,
    out_dir: str,
    read_dur: bool = False
) -> None:
    # https://github.com/JRWang-SDUST/SDUST-Dataset
    cls_list = [
        'IF0.2', 'IF0.4', 'IF0.6', 'NC', 'OF0.2', 'OF0.4',
        'OF0.6', 'RF0.2', 'RF0.4', 'RF0.6'
    ]

    os.makedirs(out_dir, exist_ok=True)
    path_list = sorted(glob.glob(os.path.join(root_dir, '**/*.wav'), recursive=True))
    info = defaultdict(list)
    for path in path_list:
        file = os.path.basename(path)
        for cls in cls_list:
            if file.startswith(cls):
                scene = cls
                break
        status = 'normal' if scene == 'NC' else 'anomaly'
        info['file'].append(file)
        info['scene'].append(scene)
        info['status'].append(status)
        info['path'].append(path)
        info['ori'].append('_'.join(file.split('_')[:-2]))
        if read_dur:
            audio_info = read_wav_info(path)
            info['dur'].append(round(audio_info['num_frames'] / audio_info['sample_rate'], 1))
    df = pd.DataFrame(info)
    df.to_csv(os.path.join(out_dir, 'all.csv'), index=False)


def gen_pu(
    root_dir: str,
    out_dir: str,
    read_dur: bool = False
) -> None:
    cls_list = [
        'healthy', 'IR', 'OR'
    ]

    os.makedirs(out_dir, exist_ok=True)
    path_list = sorted(glob.glob(os.path.join(root_dir, '**', '*.wav'), recursive=True))
    info = defaultdict(list)
    for path in path_list:
        file = os.path.basename(path)
        for cls in cls_list:
            if file.startswith(cls):
                scene = cls
                break
        info['file'].append(file)
        info['scene'].append(scene)
        info['status'].append('normal' if scene == 'healthy' else 'anomaly')
        info['path'].append(path)
        info['ori'].append('_'.join(file.split('_')[:-2]))
        if read_dur:
            audio_info = read_wav_info(path)
            info['dur'].append(round(audio_info['num_frames'] / audio_info['sample_rate'], 1))
    df = pd.DataFrame(info)
    df.to_csv(os.path.join(out_dir, 'all.csv'), index=False)


def gen_sdust_gear(
    root_dir: str,
    out_dir: str,
    read_dur: bool = False,
    num_readers: int = 4
) -> None:
    cls_list = [
        'normal', 'planetrayfracture', 'planetraypitting', 'planetraywear',
        'sunfracture', 'sunpitting', 'sunwear'
    ]
    if read_dur:
        manager = mp.Manager()
        path_queue = mp.Queue()
        info_dict = manager.dict()
        reader_list = [
            mp.Process(target=wav_reader, args=(path_queue, info_dict), daemon=True)
            for _ in range(num_readers)
        ]
        [r.start() for r in reader_list]

    os.makedirs(out_dir, exist_ok=True)
    path_list = sorted(glob.glob(os.path.join(root_dir, '*.wav')))
    info = defaultdict(list)
    for path in path_list:
        file = os.path.basename(path)
        for cls in cls_list:
            if cls in file:
                scene = cls
                break
        status = 'normal' if scene == 'normal' else 'anomaly'
        info['file'].append(file)
        info['scene'].append(scene)
        info['status'].append(status)
        info['path'].append(path)
        info['ori'].append('_'.join(file.split('_')[:-2]))
        if read_dur:
            path_queue.put(path)

    if read_dur:
        for path in path_list:
            if path not in info_dict:
                time.sleep(0.5)
            for k in info_dict[path].keys():
                info[k].append(info_dict[path][k])
        while not path_queue.empty():
            try:
                path_queue.get_nowait()
            except queue.Empty:
                break
    df = pd.DataFrame(info)
    df.to_csv(os.path.join(out_dir, 'all.csv'), index=False)

    if read_dur:
        path_queue.put(None)
        [r.join() for r in reader_list]


def gen_umged(
    root_dir: str,
    out_dir: str,
    read_dur: bool = False,
    num_readers: int = 4
) -> None:
    cls_list_G = ['G1', 'G2']
    cls_list_E = [
        'E00', 'E02', 'E04', 'E06', 'E08', 'E10',
        'E12', 'E14', 'E16', 'E18', 'E20'
    ]
    if read_dur:
        manager = mp.Manager()
        path_queue = mp.Queue()
        info_dict = manager.dict()
        reader_list = [
            mp.Process(target=wav_reader, args=(path_queue, info_dict), daemon=True)
            for _ in range(num_readers)
        ]
        [r.start() for r in reader_list]

    os.makedirs(out_dir, exist_ok=True)
    path_list = sorted(glob.glob(os.path.join(root_dir, '*.wav')))
    info = defaultdict(list)
    for path in path_list:
        file = os.path.basename(path)
        for cls in cls_list_G:
            if cls in file:
                scene_G = cls
                break
        for cls in cls_list_E:
            if cls in file:
                scene_E = cls
                break
        status = 'normal' if scene_G == 'G1' and scene_E == 'E00' else 'anomaly'
        info['file'].append(file)
        info['scene_G'].append(scene_G)
        info['scene_E'].append(scene_E)
        info['status'].append(status)
        info['path'].append(path)
        info['ori'].append('_'.join(file.split('_')[:-2]))
        if read_dur:
            path_queue.put(path)

    if read_dur:
        for path in path_list:
            if path not in info_dict:
                time.sleep(0.5)
            for k in info_dict[path].keys():
                info[k].append(info_dict[path][k])
        while not path_queue.empty():
            try:
                path_queue.get_nowait()
            except queue.Empty:
                break
    df = pd.DataFrame(info)
    df.to_csv(os.path.join(out_dir, 'all.csv'), index=False)

    if read_dur:
        path_queue.put(None)
        [r.join() for r in reader_list]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', default='conf/basic.yaml')
    parser.add_argument('--out_dir', default='data/meta_data')
    parser.add_argument('--ds', type=str, default=None)
    parser.add_argument('--read_dur', action='store_true', default=False)
    opt = parser.parse_args()

    with open(opt.conf, 'r', encoding='utf-8') as fp:
        conf = yaml.safe_load(fp)

    os.makedirs(opt.out_dir, exist_ok=True)

    if opt.ds == 'all' or opt.ds.startswith('dcase'):
        if opt.ds in ['all', 'dcase'] or opt.ds == 'dcase20':
            dcase20_out = os.path.join(opt.out_dir, 'dcase20')
            gen_20(conf['dcase20_dir'], conf['dcase_eval_gt_dir'], dcase20_out, read_dur=True)

        if opt.ds in ['all', 'dcase'] or opt.ds == 'dcase21':
            dcase21_out = os.path.join(opt.out_dir, 'dcase21')
            gen_21(conf['dcase21_dir'], conf['dcase_eval_gt_dir'], dcase21_out, read_dur=True)

        if opt.ds in ['all', 'dcase'] or opt.ds == 'dcase22':
            dcase22_out = os.path.join(opt.out_dir, 'dcase22')
            gen_222324(conf['dcase22_dir'], conf['dcase_eval_gt_dir'], dcase22_out, read_dur=True)

        if opt.ds in ['all', 'dcase'] or opt.ds == 'dcase23':
            dcase23_out = os.path.join(opt.out_dir, 'dcase23')
            gen_222324(conf['dcase23_dir'], conf['dcase_eval_gt_dir'], dcase23_out, read_dur=True)

        if opt.ds in ['all', 'dcase'] or opt.ds == 'dcase24':
            dcase24_out = os.path.join(opt.out_dir, 'dcase24')
            gen_222324(conf['dcase24_dir'], conf['dcase_eval_gt_dir'], dcase24_out, read_dur=True)

        if opt.ds in ['all', 'dcase'] or opt.ds == 'dcase25':
            dcase25_out = os.path.join(opt.out_dir, 'dcase25')
            gen_25(conf['dcase25_dir'], conf['dcase_eval_gt_dir'], dcase25_out, read_dur=True)

    elif opt.ds == 'all' or opt.ds.startswith('idmt'):
        # IDMT-ISA-COMPRESSED-AIR
        idmt_air_out = os.path.join(opt.out_dir, 'idmt_air')
        gen_idmt_isa_compressed_air(
            conf['idmt_isa_compressed_air_dir'],
            idmt_air_out,
            read_dur=True
        )

        # IDMT-ISA-ELECTRIC-ENGINE
        idmt_engine_out = os.path.join(opt.out_dir, 'idmt_engine')
        gen_idmt_isa_electric_engine(
            conf['idmt_isa_electric_engine_dir'],
            idmt_engine_out,
            read_dur=opt.read_dur
        )

    elif opt.ds == 'all' or opt.ds == 'wt_plane_gearbox':
        # WT_planetary_gearbox
        wt_plane_gearbox_out = os.path.join(opt.out_dir, 'wt_plane_gearbox')
        gen_wt_plane_gearbox(
            conf['wt_plane_gearbox_dir'],
            wt_plane_gearbox_out,
            read_dur=opt.read_dur
        )

    elif opt.ds == 'all' or opt.ds == 'mafaulda':
        # mafaulda_vibration
        mafaulda_vib_out = os.path.join(opt.out_dir, 'mafaulda_vib')
        gen_mafaulda(conf['mafaulda_vib_dir'], mafaulda_vib_out, read_dur=opt.read_dur)

        # mafaulda_sound
        mafaulda_sound_out = os.path.join(opt.out_dir, 'mafaulda_sound')
        gen_mafaulda(conf['mafaulda_sound_dir'], mafaulda_sound_out, read_dur=opt.read_dur)

    elif opt.ds == 'all' or opt.ds == 'sdust':
        # sdust_bearing
        sdust_bearing = os.path.join(opt.out_dir, 'sdust_bearing')
        gen_sdust_bearing(conf['sdust_bearing_dir'], sdust_bearing)

        # sdust_gear
        sdust_gear = os.path.join(opt.out_dir, 'sdust_gear')
        gen_sdust_gear(conf['sdust_gear_dir'], sdust_gear, read_dur=opt.read_dur)

    elif opt.ds == 'all' or opt.ds == 'umged':
        # umged_sound
        umged_sound = os.path.join(opt.out_dir, 'umged_sound')
        gen_umged(conf['umged_sound_dir'], umged_sound, read_dur=True)

        # umged_vib
        umged_vib = os.path.join(opt.out_dir, 'umged_vib')
        gen_umged(conf['umged_vib_dir'], umged_vib, read_dur=True)

        # umged_cur
        umged_cur = os.path.join(opt.out_dir, 'umged_cur')
        gen_umged(conf['umged_cur_dir'], umged_cur, read_dur=True)

        # umged_vol
        umged_vol = os.path.join(opt.out_dir, 'umged_vol')
        gen_umged(conf['umged_vol_dir'], umged_vol, read_dur=True)

    elif opt.ds == 'all' or opt.ds == 'pu':
        # pu_vibration
        pu_vib_out = os.path.join(opt.out_dir, 'pu_vib')
        gen_pu(conf['pu_vib_dir'], pu_vib_out, read_dur=opt.read_dur)

        # pu_current
        pu_cur_out = os.path.join(opt.out_dir, 'pu_cur')
        gen_pu(conf['pu_cur_dir'], pu_cur_out, read_dur=opt.read_dur)

    else:
        raise KeyError(f'Unknown dataset {opt.ds}')
