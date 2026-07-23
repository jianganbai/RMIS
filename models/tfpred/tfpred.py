import torch
import torch.nn as nn

from typing import Dict, Any

from .Models.ResNet1D import resnet18


class ModelBase(nn.Module):
    def __init__(self, dim: int = 128) -> None:
        super().__init__()
        self.net = resnet18(norm_layer=None)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class TFPred(nn.Module):
    def __init__(
        self,
        ckpt: str,
        emb_size: int = 0,
    ) -> None:
        super().__init__()
        self.model = ModelBase(dim=128)
        self._load_official_encoder(ckpt)
        if emb_size > 0 and emb_size != 128:
            self.fc = nn.Linear(128, emb_size)

    def _load_official_encoder(self, ckpt: str) -> None:
        checkpoint = torch.load(ckpt, map_location='cpu', weights_only=True)
        encoder_state = {
            k[len('encoderT.'):]: v
            for k, v in checkpoint.items()
            if k.startswith('encoderT.')
        }
        self.model.load_state_dict(encoder_state, strict=True)

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
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