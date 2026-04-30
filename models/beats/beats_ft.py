import torch
import torch.nn as nn

from copy import deepcopy
from collections import OrderedDict
from typing import Dict, Any, Optional

from .BEATs import BEATs, BEATsConfig


class BEATs_FT(nn.Module):
    def __init__(
        self,
        ckpt: str,
        MEAN: float = 15.41663,
        STD: float = 6.55582,
    ):
        super().__init__()
        checkpoint = torch.load(ckpt, weights_only=True)
        self.cfg = checkpoint['cfg']
        cfg = BEATsConfig(deepcopy(self.cfg))
        self.model = BEATs(cfg)
        self.model.load_state_dict(checkpoint['model'])

        self.backbone_dim = cfg.encoder_embed_dim
        self.MEAN = MEAN
        self.STD = STD

    def state_dict(self, **kwargs):
        state_dict = OrderedDict()
        state_dict['cfg'] = self.cfg
        state_dict['model'] = super().state_dict(**kwargs)
        return state_dict

    def load_state_dict(self, state_dict, strict: bool = False, assign: bool = False):
        if 'model' in state_dict.keys():
            missing_unexpected = super().load_state_dict(state_dict['model'], strict, assign)
        else:  # compatible with old checkpoints
            missing_unexpected = super().load_state_dict(state_dict, strict, assign)
        return missing_unexpected

    def embedding(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x, _, _ = self.model.extract_features(
            source=x,
            padding_mask=padding_mask,
            fbank_mean=self.MEAN,
            fbank_std=self.STD,
            layer=None
        )
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
