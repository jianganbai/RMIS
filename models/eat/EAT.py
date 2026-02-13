
import torch
import torch.nn as nn

from transformers import AutoModel
from typing import Dict, Any


class EAT(nn.Module):
    def __init__(
        self,
        model_id: str,
        MEAN: float = -4.2677393,
        STD: float = 4.5689974,
        **kwargs
    ) -> None:
        super().__init__()
        self.MEAN = MEAN
        self.STD = STD

        self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True)

    def embedding(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        n_frames = x.shape[1]
        if n_frames < 1024:
            x = torch.nn.ZeroPad2d((0, 0, 0, 1024 - n_frames))(x)
        else:
            x = x[:, :1024, :]
        x = (x - self.MEAN) / (self.STD * 2)  # normalize
        x = x.unsqueeze(1)  # [B, 1, T, D]

        x = self.model.extract_features(x)
        CLS = x[:, 0]
        return CLS

    def forward(
        self,
        x: torch.Tensor,
        **kwargs
    ) -> Dict[str, Any]:
        x = self.embedding(x)
        output_dict = {'embedding': x}
        return output_dict
