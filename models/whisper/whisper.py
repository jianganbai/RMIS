import torch
import torch.nn as nn

from typing import Dict, Any

from .encoder import AudioEncoder
from .audio import log_mel_spectrogram, pad_or_trim


class Whisper(nn.Module):
    def __init__(
        self,
        n_mels: int,
        n_audio_ctx: int,
        n_audio_state: int,
        n_audio_head: int,
        n_audio_layer: int,
        ckpt: str,
        feat_aggre: str = 'mean_pool',
        freeze_encoder: bool = True,
    ) -> None:
        super().__init__()
        if feat_aggre != 'mean_pool':
            raise NotImplementedError(f'Feature aggregation {feat_aggre} is not implemented.')

        self.n_mels = n_mels
        self.encoder = AudioEncoder(
            n_mels=n_mels,
            n_ctx=n_audio_ctx,
            n_state=n_audio_state,
            n_head=n_audio_head,
            n_layer=n_audio_layer,
        )
        encoder_dict = torch.load(ckpt, map_location='cpu', weights_only=True)
        self.encoder.load_state_dict(encoder_dict)

        if freeze_encoder:
            self.encoder.eval()
            self.encoder.requires_grad_(False)

    def process_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        audio = pad_or_trim(waveform)
        return log_mel_spectrogram(audio, n_mels=self.n_mels)

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        log_mel = self.process_audio(x)
        x, _ = self.encoder(log_mel)
        return torch.mean(x, dim=1)

    def forward(
        self,
        x: torch.Tensor,
        **kwargs
    ) -> Dict[str, Any]:
        x = self.embedding(x)
        output_dict = {'embedding': x}
        return output_dict