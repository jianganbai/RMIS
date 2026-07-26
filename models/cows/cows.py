import torch
import torch.nn as nn

from typing import Dict, Any

from .Model.ResNet1D import resnet50_1D


class CoWS(nn.Module):
    def __init__(
        self,
        ckpt: str,
        emb_size: int = 0,
    ) -> None:
        super().__init__()
        self.model = resnet50_1D()
        self.model.fc = nn.Identity()
        self._load_ckpt(ckpt)

        if emb_size > 0 and emb_size != 512:
            self.fc = nn.Linear(512, emb_size)

    def _load_ckpt(self, ckpt: str) -> None:
        checkpoint = torch.load(ckpt, map_location='cpu', weights_only=False)
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']
            elif 'model' in checkpoint:
                checkpoint = checkpoint['model']

        state_dict = {}
        for key, value in checkpoint.items():
            key = key.replace('module.', '').replace('model.', '')
            if key.startswith('fc.'):
                continue
            state_dict[key] = value
        self.model.load_state_dict(state_dict, strict=False)

    def embedding(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.model(x)
        if hasattr(self, 'fc'):
            x = self.fc(x)
        return x

    def forward(
        self,
        x: torch.Tensor,
        **kwargs
    ) -> Dict[str, Any]:
        x = self.embedding(x)
        output_dict = {'embedding': x}
        return output_dict
