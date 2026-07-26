import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, Any

from .models.passt import get_model
from .models.preprocess import AugmentMelSTFT


# RMIS adapter for PaSST. The imported models package contains the
# backbone and preprocessing implementation reused by this wrapper.
class PaSST(nn.Module):
    def __init__(
        self,
        ckpt: str,
        sample_rate: int = 32000,
        n_mels: int = 128,
        input_tdim: int = 998,
    ) -> None:
        super().__init__()
        self.input_tdim = input_tdim
        self.mel_transform = AugmentMelSTFT(n_mels=n_mels, sr=sample_rate)

        self.encoder = get_model(
            arch='passt_s_swa_p16_128_ap476',
            pretrained=False,
            n_classes=527,
            in_channels=1,
            fstride=10,
            tstride=10,
            input_fdim=128,
            input_tdim=input_tdim,
            u_patchout=0,
            s_patchout_t=0,
            s_patchout_f=0,
        )

        state_dict = torch.load(ckpt, map_location='cpu')
        missing_unexpected = self.encoder.load_state_dict(state_dict, strict=False)
        print(missing_unexpected)

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mel_transform(x)
        x = x.unsqueeze(1)

        if x.shape[-1] > self.input_tdim:
            x = x[..., :self.input_tdim]
        elif x.shape[-1] < self.input_tdim:
            x = F.pad(x, (0, self.input_tdim - x.shape[-1]))

        _, features = self.encoder(x)
        return features

    def forward(
        self,
        x: torch.Tensor,
        **kwargs
    ) -> Dict[str, Any]:
        x = self.embedding(x)
        output_dict = {'embedding': x}
        return output_dict
