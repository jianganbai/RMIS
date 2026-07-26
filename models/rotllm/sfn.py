import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, Any, Optional

from .models.SFN import SpecFoldNet


# RMIS adapter for RotLLM SpecFoldNet checkpoints. The imported models
# package keeps the network implementation separate from RMIS preprocessing.
def dcn(signal_1d: torch.Tensor, n_f: int = 24000, beta: float = 0.01) -> torch.Tensor:
    if signal_1d.ndim == 1:
        signal_1d = signal_1d.unsqueeze(0)

    x = signal_1d.to(dtype=torch.float64)
    n = x.shape[-1]
    if n == 0:
        coeff = x.new_zeros((x.shape[0], n_f))
    else:
        v = torch.cat([x[..., ::2], x[..., 1::2].flip(dims=[-1])], dim=-1)
        spectrum = torch.fft.fft(v, dim=-1)
        k = torch.arange(n, device=x.device, dtype=x.dtype)
        angles = -math.pi * k / (2 * n)
        twiddle = torch.complex(torch.cos(angles), torch.sin(angles))
        coeff = 2.0 * (spectrum[..., :n] * twiddle).real
        if coeff.shape[-1] < n_f:
            coeff = F.pad(coeff, (0, n_f - coeff.shape[-1]))
        else:
            coeff = coeff[..., :n_f]

    coeff = coeff * torch.sqrt(n_f / (coeff.pow(2).sum(dim=-1, keepdim=True) + 1e-12))
    return (coeff * beta).to(dtype=torch.float32)


class SFN(nn.Module):
    def __init__(
        self,
        ckpt: str,
        emb_size: int = 0,
    ) -> None:
        super().__init__()
        self.model = SpecFoldNet()
        weights = torch.load(ckpt, map_location='cpu')
        self.model.encoder.load_state_dict(weights, strict=True)

        if emb_size > 0 and emb_size != self.model.encode_len:
            self.fc = nn.Linear(self.model.encode_len, emb_size)

    def embedding(
        self,
        x: torch.Tensor,
        valid_len: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        n_f = self.model.fold_num * self.model.sub_spec_len
        if valid_len is None:
            dct_arr = dcn(x, n_f=n_f)
        else:
            valid_len = valid_len.to(device=x.device, dtype=torch.long).clamp_(0, x.size(1))
            dct_arr = x.new_zeros((x.size(0), n_f), dtype=torch.float32)
            for length in torch.unique(valid_len, sorted=True):
                mask = valid_len == length
                current_len = int(length.item())
                if current_len == 0:
                    continue
                dct_arr[mask] = dcn(x[mask, :current_len], n_f=n_f)

        x = dct_arr.to(device=x.device, dtype=x.dtype)
        x = x.view(x.size(0), self.model.fold_num, self.model.sub_spec_len)
        x = self.model.encoder(x)
        x = x.view(x.size(0), self.model.encode_len)
        if hasattr(self, 'fc'):
            x = self.fc(x)
        return x

    def forward(
        self,
        x: torch.Tensor,
        valid_len: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        x = self.embedding(x, valid_len)
        output_dict = {'embedding': x}
        return output_dict
