
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, Any

from .models.fisher import FISHER


class FISHER_infer(nn.Module):
    def __init__(
        self,
        ckpt: str,
        MEAN: float = 0.0,
        STD: float = 0.5,
    ) -> None:
        super().__init__()
        self.MEAN = MEAN
        self.STD = STD

        self.model = FISHER.from_pretrained(ckpt)
        self.band_width = self.model.cfg.band_width

    def fix_spec_size(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        # time-wise
        if x.shape[-2] < 1024:
            x = F.pad(x, (0, 0, 0, 1024 - x.shape[1]))
        else:
            x = x[:, :1024]

        # freq-wise
        if x.shape[-1] < self.band_width:
            x = F.pad(x, (0, self.band_width - x.shape[2]))
        return x

    def embedding(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        # x: [B, T, D]
        x = (x - self.MEAN) / (self.STD * 2)  # normalize
        x = x.unsqueeze(1)  # [B, 1, T, D]

        x = self.model.extract_features(x)
        return x

    def forward(
        self,
        x: torch.Tensor,
        **kwargs
    ) -> Dict[str, Any]:
        x = self.fix_spec_size(x)
        x = self.embedding(x)
        output_dict = {'embedding': x}
        return output_dict
