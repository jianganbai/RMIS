import torch
import warnings
import torch.nn as nn

from typing import Literal, Dict, Any

from .dasheng.pretrained.pretrained import (
    dasheng_base, dasheng_06B, dasheng_12B
)


class Dasheng(nn.Module):
    def __init__(
        self,
        scale: Literal['base', '0.6B', '1.2B'] = 'base',
        mask_ratio: float = 0.0,
    ) -> None:
        super().__init__()
        if scale == 'base':
            self.model = dasheng_base()  # 768
        elif scale == '0.6B':
            self.model = dasheng_06B()  # 1280
        elif scale == '1.2B':
            self.model = dasheng_12B()  # 1536
        else:
            raise KeyError(f'Invalid scale {scale} for Dasheng')

        if mask_ratio > 0.0:
            warnings.warn('Inner built mask should be disabled.')

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        x = torch.mean(x, dim=1)
        return x.squeeze(1)

    def forward(
        self,
        x: torch.Tensor,
        **kwargs
    ) -> Dict[str, Any]:
        x = self.embedding(x)
        output_dict = {'embedding': x}
        return output_dict
