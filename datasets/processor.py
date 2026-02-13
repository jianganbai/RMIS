import torch

from typing import Literal, Optional


class Normalizer:
    def __init__(
        self,
        norm_type: Literal['all', 'temporal'] = 'temporal',
        MEAN: Optional[float] = None,
        STD: Optional[float] = None,
        coeff: float = 1.0
    ) -> None:
        self.norm_type = norm_type
        self.MEAN = MEAN
        self.STD = STD
        self.coeff = coeff
        assert self.norm_type in ['all', 'temporal'], \
            f'Unknown normalization {norm_type}'

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: [1, T, D]
        if self.norm_type == 'all':
            if self.MEAN is None or self.STD is None:
                x = (x - x.mean()) / (x.std() * self.coeff + 1e-10)
            else:
                x = (x - self.MEAN) / (self.STD * self.coeff)

        elif self.norm_type == 'temporal':
            # no pre-defined stats
            if len(x.shape) == 3:  # fbank
                MEAN, STD = x.mean(dim=-2, keepdim=True), x.std(dim=-2, keepdim=True)
            else:  # waveform
                MEAN, STD = x.mean(), x.std()
            x = (x - MEAN) / (STD * self.coeff + 1e-10)

        return x
