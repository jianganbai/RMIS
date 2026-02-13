import os
import glob
import copy
import torch
import random
import warnings
import soundfile
import torchaudio
import pandas as pd
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import Dataset
from typing import (
    List, Dict, Any,
    Tuple, Optional, Union
)

from .spec import build_stft, build_fbank
from .processor import Normalizer


def read_data(
    args: Dict[str, Any],
    df: pd.DataFrame,
    csv: Optional[str] = None
) -> List[torch.Tensor]:
    input_sr: int = args['sample_rate']
    data_sr: int = args.get('data_sr', input_sr)
    data_sr_conf = args.get('data_sr_conf', {})
    data_sr_resample = data_sr_conf.get('resample', 'torchaudio')
    # interval: equal interval sampling, with aliasing
    # torchaudio: use torchaudio resample, low pass filter + equal interval sampling
    assert data_sr_resample in ['interval', 'torchaudio'], \
        f'Unknown resampling metric {data_sr_resample} for data_sr'
    if data_sr_resample == 'interval':
        # _, sr = soundfile.read(df.loc[0, 'path'])
        sr = soundfile.info(df.loc[0, 'path']).samplerate
        if sr % data_sr != 0:
            warnings.warn(
                f"Original sr {sr} is not a multiplier of desired data_sr {data_sr}."
                'Default to torchaudio resampling.'
            )
            data_sr_resample = 'torchaudio'

    use_shard = args.get('use_shard', False)
    if use_shard:
        shard_conf = args.get('shard_conf', {})
        num_shards = shard_conf.get('num_shards', 1)
        if args['downstream'] == 'dcase':
            assert csv is not None
            shard_dir = os.path.join(
                shard_conf['shard_dir'],
                os.path.splitext(os.path.relpath(csv, args['meta_data_dir']))[0]
            )
        elif args['downstream'] in [
            'idmt_engine', 'gwbw', 'jy_wind_turbine_sound', 'jy_wind_turbine_vib',
            'chem_fault_transfer'
        ] or args['downstream'].endswith('_pred'):
            assert csv is not None
            shard_dir = os.path.join(
                shard_conf['shard_dir'], args['downstream'],
                os.path.splitext(os.path.basename(csv))[0],
            )
        else:
            shard_dir = os.path.join(
                shard_conf['shard_dir'], args['downstream']
            )
        os.makedirs(shard_dir, exist_ok=True)
        shard_files = sorted(
            glob.glob(os.path.join(shard_dir, f'{data_sr_resample}_{data_sr}_[0-9]*.pt')),
            key=lambda x: int(x.split('_')[-1][:-3])
        )

    data = []
    if (not use_shard) or len(shard_files) == 0:
        last_f = None
        for i, f in tqdm(enumerate(df['path']), total=len(df)):
            if f.endswith('csv'):
                # additional keys in conf: with_header, (data_sr)
                # additional keys in meta_data: ch, (sr)
                if last_f != f:
                    header = 'infer' if args.get('with_header', True) else None
                    try:
                        xdf = pd.read_csv(f, header=header, engine='pyarrow')
                    except Exception:
                        if i == 0:
                            warnings.warn('Recommend to install pyarrow for faster csv reading!')
                        xdf = pd.read_csv(f, header=header)
                if isinstance(df.loc[i, 'ch'], str):
                    ch = df.loc[i, 'ch']
                else:
                    ch = xdf.columns[df.loc[i, 'ch'].item()]
                x = xdf[ch].to_numpy()
                sr = int(df.loc[i, 'sr']) if 'sr' in df.columns else data_sr
            else:
                x, sr = soundfile.read(f)  # (num_frame) or (num_frame, num_channel)
                assert x.ndim == 1, 'Only mono channel is supported!'
            x = torch.from_numpy(x).unsqueeze(0).to(torch.float32)
            if sr != data_sr:
                if data_sr_resample == 'interval' and sr % data_sr == 0:
                    x = x[:, ::(sr // data_sr)].clone()
                else:
                    x = torchaudio.transforms.Resample(
                        orig_freq=sr, new_freq=data_sr
                    )(x)

            data.append(x)
            last_f = f
        if use_shard:
            clip_per_shard = len(data) // num_shards + 1
            for i in range(num_shards):
                start = i * clip_per_shard
                stop = min((i + 1) * clip_per_shard, len(data))
                save_path = os.path.join(shard_dir, f'{data_sr_resample}_{data_sr}_{i}.pt')
                torch.save(
                    {df.loc[i, 'file']: data[i] for i in range(start, stop)},
                    save_path
                )
                print(f'Saving shard {save_path}')
    else:
        assert len(shard_files) == num_shards, \
            f'Expects {num_shards} shards, but reads {len(shard_files)} shards'
        print(
            'Reading data from shards {}. Data_sr: {}; Input_sr: {}; Data_sr resample: {}'
            .format(shard_files, data_sr, input_sr, data_sr_resample)
        )
        shard_data = {}
        for i in range(num_shards):
            shard_data.update(torch.load(shard_files[i], weights_only=True))
        for file in df['file']:
            data.append(shard_data[file])
        assert len(data) == len(df), \
            f'Expects {len(df)} samples, but reads {len(data)} samples from shards'

    if data_sr != input_sr:
        for i in range(len(data)):
            data[i] = torchaudio.transforms.Resample(
                orig_freq=data_sr, new_freq=input_sr
            )(data[i])

    return data


class Raw_Dataset(Dataset):
    def __init__(
        self,
        ds: str,
        csv: Union[str, pd.DataFrame],
        args: Dict[str, Any],
        hop_size: Optional[int] = None,
        desc: Optional[str] = None,
        pre_read_data: Optional[List[torch.Tensor]] = None
    ) -> None:
        super().__init__()
        self.ds = ds
        self.input_sr = args['sample_rate']
        self.data_sr = args.get('data_sr', self.input_sr)
        self.win_frames = args['win_frames']
        self.hop_size = hop_size
        self.desc = desc
        self.num_classes = -1
        self.map_ = {}  # {label: label_id}
        self.comple = args.get('comple', 'pad')
        self.dt = 'train' if hop_size is None else 'test'
        self.args = copy.deepcopy(args)
        self.norm = args.get('norm', False)
        if self.norm:
            self.normalizer = Normalizer(**args.get('norm_conf', {}))

        if ds.startswith('dcase'):
            assert isinstance(csv, str)
            self.mt = os.path.basename(csv).split('.')[0]
        if isinstance(csv, str):
            self.df = pd.read_csv(csv, index_col=None)
        else:
            self.df = csv.copy()

        if pre_read_data is not None:
            self.data = pre_read_data
        else:
            self.data = read_data(
                args, self.df,
                csv=csv if isinstance(csv, str) else None
            )

        self.__segmentation__()

    def __segmentation__(self):
        self.index_map = {}
        self.seg_weights = []
        ctr = 0
        for i in range(len(self.data)):
            num_frames = self.data[i].shape[-1]
            if num_frames <= self.win_frames:
                self.index_map[ctr] = (i, 0)
                ctr += 1
            elif self.hop_size:
                j = 0
                while j + self.win_frames <= num_frames:
                    self.index_map[ctr] = (i, j)
                    ctr += 1
                    j += self.hop_size
                # only save remainder if it is no less than 5% of the window size
                if j + 0.05 * self.win_frames <= num_frames:
                    self.index_map[ctr] = (i, num_frames - self.win_frames)
                    ctr += 1
            else:
                for _ in range(num_frames // self.win_frames):
                    self.index_map[ctr] = (i, None)  # sample on the fly
                    ctr += 1
        self.length = ctr

    def __getwav__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        file_idx, offset = self.index_map[idx]
        info = self.df.iloc[file_idx].to_dict()
        for delete_key in ['path', 'ori']:
            if delete_key in info.keys():
                del info[delete_key]

        if offset is None:  # train
            num_frames = self.data[file_idx].shape[-1]
            offset = random.randrange(0, num_frames - self.win_frames)
            wav = self.data[file_idx][0, offset: offset+self.win_frames].to(torch.float32)

        else:  # test
            # can exceed the actual length
            wav = self.data[file_idx][0, offset: offset + self.win_frames].to(torch.float32)

        # wav: 1d
        return wav, info

    def __pad_wav__(self, wav: torch.Tensor) -> Tuple[torch.Tensor, int]:
        # wav: 1d
        if self.comple == 'cycle':
            valid_len = self.win_frames
            if wav.shape[-1] < self.win_frames:
                wav = wav.repeat(self.win_frames // wav.shape[-1] + 1)
            wav = wav[:self.win_frames]
        else:  # pad
            if wav.shape[-1] < self.win_frames:
                valid_len = wav.shape[-1]
                wav = F.pad(wav, (0, self.win_frames - wav.shape[-1]), value=0)
            else:
                valid_len = self.win_frames
                wav = wav[:self.win_frames]
        return wav, valid_len

    def __len__(self):
        return self.length

    def __getitem__(self, idx) -> Any:
        pass


class Audio_Dataset(Raw_Dataset):
    def __init__(
        self,
        ds: str,
        csv: Union[str, pd.DataFrame],
        args: Dict[str, Any],
        hop_size: Optional[int] = None,
        desc: Optional[str] = None,
        pre_read_data: Optional[List[torch.Tensor]] = None
    ) -> None:
        super().__init__(
            ds=ds,
            csv=csv,
            args=args,
            hop_size=hop_size,
            desc=desc,
            pre_read_data=pre_read_data
        )

        self.feat_type = args['feat_type']
        assert self.feat_type in ['wav', 'stft', 'fbank'], \
            f'Unsupported feature {self.feat_type}'
        if self.feat_type == 'stft':
            stft_conf = args.get('stft_conf', {})
            stft_conf['sample_rate'] = self.input_sr
            self.cal_stft = build_stft(stft_conf)

        elif self.feat_type == 'fbank':
            fbank_conf = args.get('fbank_conf', {})
            fbank_conf['sample_rate'] = self.input_sr
            self.cal_fbank = build_fbank(fbank_conf)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        wav, info = self.__getwav__(idx)
        wav = wav - wav.mean()
        wav, valid_len = self.__pad_wav__(wav)

        if self.feat_type == 'stft':
            wav, valid_len = self.cal_stft(wav, valid_len)  # [time, freq]
            if self.norm:
                wav = self.normalizer(wav)

        elif self.feat_type == 'fbank':
            wav, valid_len = self.cal_fbank(wav, valid_len)  # [time, freq]
            if self.norm:
                wav = self.normalizer(wav)

        info['x'] = wav
        info['valid_len'] = valid_len
        return info


class Audio_DatasetNG(Dataset):
    def __init__(
        self,
        ds: str,
        csv: Union[str, pd.DataFrame],
        args: Dict[str, Any],
        hop_size: Optional[int] = None,
        desc: Optional[str] = None,
        **kwargs
    ) -> None:
        super().__init__()
        self.ds = ds
        self.csv = csv
        self.input_sr = args['sample_rate']
        self.data_sr = args.get('data_sr', self.input_sr)
        self.win_frames = args['win_frames']
        self.hop_size = hop_size
        self.desc = desc
        self.num_classes = -1
        self.map_ = {}  # {label: label_id}
        self.comple = args.get('comple', 'pad')
        self.dt = 'train' if hop_size is None else 'test'
        self.args = copy.deepcopy(args)

        if ds.startswith('dcase'):
            assert isinstance(csv, str)
            self.mt = os.path.basename(csv).split('.')[0]
        if isinstance(csv, str):
            self.df = pd.read_csv(csv, index_col=None)
        else:
            self.df = csv.copy()

        self.__segmentation__()

        self.feat_type = args['feat_type']
        assert self.feat_type in ['wav', 'stft', 'fbank'], \
            f'Unsupported feature {self.feat_type}'
        if self.feat_type == 'stft':
            stft_conf = args.get('stft_conf', {})
            stft_conf['sample_rate'] = self.input_sr
            self.cal_stft = build_stft(stft_conf)

        elif self.feat_type == 'fbank':
            fbank_conf = args.get('fbank_conf', {})
            fbank_conf['sample_rate'] = self.input_sr
            self.cal_fbank = build_fbank(fbank_conf)

    def __segmentation__(self):
        self.index_map = {}
        self.seg_weights = []
        ctr = 0
        for i in range(len(self.df)):
            info = self.df.iloc[i].to_dict()
            if info['sample_rate'] == 0.0 or info['num_frames'] == 0.0:
                warnings.warn(f'Empty clip:\n{info}')
                continue
            num_frames = int(
                (self.input_sr / info['sample_rate'])
                * info['num_frames']
            )
            if num_frames <= self.win_frames:
                self.index_map[ctr] = (i, 0)
                ctr += 1
            elif self.hop_size:
                j = 0
                while j + self.win_frames <= num_frames:
                    self.index_map[ctr] = (i, j)
                    ctr += 1
                    j += self.hop_size
                # only save remainder if it is no less than 5% of the window size
                if j + 0.05 * self.win_frames <= num_frames:
                    self.index_map[ctr] = (i, num_frames - self.win_frames)
                    ctr += 1
            else:
                for _ in range(int(num_frames / self.win_frames)):
                    self.index_map[ctr] = (i, None)  # sample on the fly
                    ctr += 1
        self.length = ctr

    def __getwav__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        file_idx, offset = self.index_map[idx]
        info = self.df.iloc[file_idx].to_dict()

        wav, sr = soundfile.read(info['path'])
        if wav.ndim == 2:
            wav = wav.mean(1)
        wav = torch.from_numpy(wav).unsqueeze(0).to(torch.float32)
        if sr != self.input_sr:
            wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.input_sr)(wav)

        if offset is None:  # train
            num_frames = int((
                (self.input_sr / self.df.loc[file_idx, 'sample_rate'])
                * self.df.loc[file_idx, 'num_frames']
            ).item())
            offset = random.randrange(0, num_frames - self.win_frames)
            wav = wav[0, offset: offset+self.win_frames]

        else:  # test
            # can exceed the actual length
            wav = wav[0, offset: offset + self.win_frames].to(torch.float32)

        for delete_key in ['path', 'ori']:
            if delete_key in info.keys():
                del info[delete_key]

        # wav: 1d
        return wav, info

    def __pad_wav__(self, wav: torch.Tensor) -> Tuple[torch.Tensor, int]:
        # wav: 1d
        if self.comple == 'cycle':
            valid_len = self.win_frames
            if wav.shape[-1] < self.win_frames:
                wav = wav.repeat(self.win_frames // wav.shape[-1] + 1)
            wav = wav[:self.win_frames]
        else:  # pad
            if wav.shape[-1] < self.win_frames:
                valid_len = wav.shape[-1]
                wav = F.pad(wav, (0, self.win_frames - wav.shape[-1]), value=0)
            else:
                valid_len = self.win_frames
                wav = wav[:self.win_frames]
        return wav, valid_len

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        wav, info = self.__getwav__(idx)
        wav = wav - wav.mean()
        wav, valid_len = self.__pad_wav__(wav)

        if self.feat_type == 'stft':
            wav, valid_len = self.cal_stft(wav, valid_len)  # [time, freq]

        elif self.feat_type == 'fbank':
            wav, valid_len = self.cal_fbank(wav, valid_len)  # [time, freq]

        info['x'] = wav
        info['valid_len'] = valid_len
        return info
