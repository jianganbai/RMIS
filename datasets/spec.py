import torch
import librosa
import nnAudio
import nnAudio.features
import torchaudio
import numpy as np
import torch.nn as nn

from functools import partial
from typing import Any, Dict, Tuple
from torchaudio.compliance import kaldi
from transformers.audio_utils import spectrogram as hf_spectrogram
from transformers.audio_utils import window_function, mel_filter_bank


class base_stft:
    def __init__(
        self,
        stft_conf: Dict[str, Any]
    ) -> None:
        self.sr = stft_conf.get('sample_rate', 16000)
        self.n_fft = int(self.sr * stft_conf.get('frame_length', 25) / 1000)
        self.hop_length = int(self.sr * stft_conf.get('frame_shift', 10) / 1000)
        self.log = stft_conf.get('log', True)

    def __call__(self) -> Any:
        pass

    def cal_valid_len(
        self,
        x: torch.Tensor,  # [..., time ,freq]
        valid_len: int
    ) -> int:
        valid_len = min((valid_len - self.n_fft) // self.hop_length + 1, x.shape[-2])
        return valid_len


class torchaudio_stft(base_stft):
    def __init__(
        self,
        stft_conf: Dict[str, Any]
    ) -> None:
        super().__init__(stft_conf)
        self.data_type = stft_conf.get('data_type', 'magnitude')
        self.cal_stft = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            win_length=None,
            hop_length=self.hop_length,
            power=None if self.data_type == 'complex' else 1,
            center=stft_conf.get('center', False)
        )

    def __call__(
        self,
        x: torch.Tensor,
        valid_len: int
    ) -> Tuple[torch.Tensor, int]:
        '''Calculate STFT based on torchaudio

        Args:
            x (torch.Tensor): raw waveform
            valid_len: valid_len for waveform

        Returns:
            Tuple[torch.Tensor, int]: (fbank, valid_len for stft), stft: [T, D]
        '''
        if x.ndim == 1:
            x = x.unsqueeze(0)
        x = self.cal_stft(x)

        if self.data_type == 'complex':
            x = torch.concat([x.real, x.imag], dim=0)
        if self.log:
            x = torch.log(x + 1e-10)

        x = x.squeeze().transpose(-2, -1)  # [time, freq]
        valid_len = self.cal_valid_len(x, valid_len)
        return x, valid_len


class librosa_stft(base_stft):
    def __init__(
        self,
        stft_conf: Dict[str, Any]
    ) -> None:
        super().__init__(stft_conf)
        self.cal_stft = partial(
            librosa.stft,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )

    def __call__(
        self,
        x: torch.Tensor,
        valid_len: int
    ) -> Tuple[torch.Tensor, int]:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        x = self.cal_stft(x.numpy())
        x = torch.from_numpy(np.abs(x))
        if self.log:
            x = torch.log(x + 1e-10)
        x = x.squeeze().transpose(-2, -1)  # [time, freq]
        valid_len = self.cal_valid_len(x, valid_len)
        return x, valid_len


def build_stft(stft_conf: Dict[str, Any]) -> base_stft:
    stft_impl = stft_conf.get('impl', 'torchaudio')
    if stft_impl == 'torchaudio':
        cal_stft = torchaudio_stft(
            stft_conf=stft_conf,
        )
    elif stft_impl == 'librosa':
        cal_stft = librosa_stft(
            stft_conf=stft_conf,
        )
    else:
        raise NotImplementedError(f'Unsupported stft method {stft_impl}!')
    return cal_stft


class base_fbank:
    def __init__(
        self,
        fbank_conf: Dict[str, Any],
    ) -> None:
        self.sr = fbank_conf.get('sample_rate', 16000)
        self.n_fft = fbank_conf.get(
            'n_fft',
            int(self.sr * fbank_conf.get('frame_length', 25) / 1000)
        )
        self.hop_length = fbank_conf.get(
            'hop_length',
            int(self.sr * fbank_conf.get('frame_shift', 10) / 1000)
        )

    def __call__(self) -> None:
        pass

    def cal_valid_len(
        self,
        x: torch.Tensor,  # [..., time, freq]
        valid_len: int
    ) -> int:
        valid_len = min((valid_len - self.n_fft) // self.hop_length + 1, x.shape[-2])
        return valid_len


class kaldi_fbank(base_fbank):
    def __init__(
        self,
        fbank_conf: Dict[str, Any],
    ) -> None:
        super().__init__(fbank_conf)
        self.shift_15 = fbank_conf.get('shift_15', False)
        self.cal_fbank = partial(
            kaldi.fbank,
            num_mel_bins=fbank_conf.get('num_mel_bins', 80),
            frame_length=fbank_conf.get('frame_length', 25),
            frame_shift=fbank_conf.get('frame_shift', 10),
            dither=fbank_conf.get('dither', 0.0),
            sample_frequency=self.sr,
            htk_compat=fbank_conf.get('htk_compat', False),
            window_type=fbank_conf.get('window_type', 'hanning'),
        )

    def __call__(
        self,
        x: torch.Tensor,
        valid_len: int
    ) -> Tuple[torch.Tensor, int]:
        '''Calculate fbank based on kaldi

        Args:
            x (torch.Tensor): raw waveform
            valid_len: valid_len for waveform

        Returns:
            Tuple[torch.Tensor, int]: (fbank, valid_len for fbank), fbank: [T, D]
        '''
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if self.shift_15:
            x = x * (2 ** 15)
        x = self.cal_fbank(waveform=x)
        x = x.squeeze()
        valid_len = self.cal_valid_len(x, valid_len)
        return x, valid_len


class torchaudio_fbank(base_fbank):
    def __init__(
        self,
        fbank_conf: Dict[str, Any],
    ) -> None:
        super().__init__(fbank_conf)
        self.cal_fbank = nn.Sequential(
            torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sr,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=fbank_conf.get('num_mel_bins', 80),
                f_min=fbank_conf.get('f_min', 0),
                f_max=fbank_conf.get('f_max', None),
                center=fbank_conf.get('center', True),
            ),
            torchaudio.transforms.AmplitudeToDB(
                top_db=fbank_conf.get('top_db', None)
            )
        )

    def __call__(
        self,
        x: torch.Tensor,
        valid_len: int
    ) -> Tuple[torch.Tensor, int]:
        '''Calculate fbank based on torchaudio

        Args:
            x (torch.Tensor): raw waveform
            valid_len: valid_len for waveform

        Returns:
            Tuple[torch.Tensor, int]: (fbank, valid_len for fbank), fbank: [T, D]
        '''
        if x.ndim == 1:
            x = x.unsqueeze(0)
        x = self.cal_fbank(x)  # (channel, n_mels, time)
        x = x.squeeze().transpose(-2, -1)  # time first
        valid_len = self.cal_valid_len(x, valid_len)
        return x, valid_len


class nnaudio_fbank(base_fbank):
    def __init__(
        self,
        fbank_conf: Dict[str, Any]
    ) -> None:
        super().__init__(fbank_conf)
        self.cal_melspec = nnAudio.features.MelSpectrogram(
            sr=self.sr,
            n_fft=self.n_fft,
            win_length=fbank_conf.get('win_length', None),
            hop_length=self.hop_length,
            n_mels=fbank_conf.get('num_mel_bins', 80),
            fmin=fbank_conf.get('f_min', 0),
            fmax=fbank_conf.get('f_max', None),
            center=True,
            power=2,
            verbose=False
        )

    def __call__(
        self,
        x: torch.Tensor,
        valid_len: int,
    ) -> Tuple[torch.Tensor, int]:
        '''Calculate fbank based on nnAudio

        Args:
            x (torch.Tensor): raw waveform
            valid_len: valid_len for waveform

        Returns:
            Tuple[torch.Tensor, int]: (fbank, valid_len for fbank), fbank: [T, D]
        '''
        if x.ndim == 1:
            x = x.unsqueeze(0)
        x = self.cal_melspec(x)
        x = (x + torch.finfo().eps).log()
        x = x.squeeze().transpose(-2, -1)  # time first
        valid_len = self.cal_valid_len(x, valid_len)
        return x, valid_len


class huggingface_fbank(base_fbank):
    def __init__(
        self,
        fbank_conf: Dict[str, Any],
    ) -> None:
        super().__init__(fbank_conf)
        self.mel_filters = mel_filter_bank(
            num_frequency_bins=(self.n_fft >> 1) + 1,
            num_mel_filters=fbank_conf['num_mel_bins'],
            min_frequency=fbank_conf.get('min_frequency', 0),
            max_frequency=fbank_conf.get('max_frequency', 14_000),
            sampling_rate=self.sr,
            norm=fbank_conf.get('norm', None),
            mel_scale=fbank_conf.get('mel_scale', 'htk'),
        )
        self.cal_fbank = partial(
            hf_spectrogram,
            window=window_function(self.n_fft, fbank_conf.get('window_type', 'hann')),
            frame_length=self.n_fft,
            hop_length=self.hop_length,
            power=2.0,
            mel_filters=self.mel_filters,
            log_mel='dB'
        )

    def __call__(
        self,
        x: torch.Tensor,
        valid_len: int
    ) -> Tuple[torch.Tensor, int]:
        '''Calculate fbank based on Huggingface API

        Args:
            x (torch.Tensor): raw waveform
            valid_len: valid_len for waveform

        Returns:
            Tuple[torch.Tensor, int]: (fbank, valid_len for fbank), fbank: [T, D]
        '''
        y = x.numpy().squeeze()
        y = self.cal_fbank(waveform=y)
        y = torch.from_numpy(y).squeeze().transpose(-2, -1)  # time first
        valid_len = self.cal_valid_len(y, valid_len)
        return y, valid_len


def build_fbank(fbank_conf: Dict[str, Any]) -> base_fbank:
    fbank_impl = fbank_conf.get('impl', 'kaldi')
    if fbank_impl == 'kaldi':
        cal_fbank = kaldi_fbank(
            fbank_conf=fbank_conf,
        )
    elif fbank_impl == 'torchaudio':
        cal_fbank = torchaudio_fbank(
            fbank_conf=fbank_conf,
        )
    elif fbank_impl == 'nnaudio':
        cal_fbank = nnaudio_fbank(
            fbank_conf=fbank_conf,
        )
    elif fbank_impl == 'huggingface':
        cal_fbank = huggingface_fbank(
            fbank_conf=fbank_conf,
        )
    else:
        raise KeyError(f'Unsupported fbank method {fbank_impl}!')
    return cal_fbank
