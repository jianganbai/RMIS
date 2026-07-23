import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, Any, Literal, Optional

from .msd.muq import MuQ


# RMIS adapter for MuQ MSD variants. The imported msd package keeps the
# model implementation separate from the benchmark-facing wrapper.
class MuQMSD(nn.Module):
    def __init__(
        self,
        model_id: str = 'OpenMuQ/MuQ-large-msd-iter',
        feat_aggre: Literal['mean_pool'] = 'mean_pool',
        weighted_sum: bool = False,
        feat_layer: int = 12,
        emb_size: int = 0,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        if feat_aggre != 'mean_pool':
            raise ValueError(f"Unsupported feat_aggre: {feat_aggre}")
        if feat_layer < 0 or feat_layer > 12:
            raise ValueError(f"feat_layer should be in [0, 12], got {feat_layer}")

        self.model = MuQ.from_pretrained(model_id)
        self.weighted_sum = weighted_sum
        self.feat_layer = feat_layer

        if weighted_sum:
            self.feat_weight = nn.Parameter(torch.zeros(13, dtype=torch.float32))
        if emb_size > 0 and emb_size != 1024:
            self.fc = nn.Linear(1024, emb_size)

        if freeze:
            self.model.eval()
            self.model.requires_grad_(False)

    @staticmethod
    def _make_attention_mask(valid_len: torch.Tensor, length: int) -> torch.Tensor:
        idx = torch.arange(length, device=valid_len.device).unsqueeze(0)
        return (idx < valid_len.unsqueeze(1)).to(torch.long)

    def layers_weighted_sum(self, x: torch.Tensor) -> torch.Tensor:
        norm_weight = F.softmax(self.feat_weight, dim=-1).view(-1, 1, 1, 1)
        return torch.sum(norm_weight * x, dim=0)

    def embedding(
        self,
        x: torch.Tensor,
        valid_len: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attention_mask = None
        if valid_len is not None:
            attention_mask = self._make_attention_mask(valid_len, x.shape[1])

        outputs = self.model(x, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        if self.weighted_sum:
            x = self.layers_weighted_sum(torch.stack(hidden_states, dim=0))
        else:
            x = hidden_states[self.feat_layer]

        x = x.mean(dim=1)
        if hasattr(self, 'fc'):
            x = self.fc(x)
        return x

    def forward(
        self,
        x: torch.Tensor,
        valid_len: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        x = self.embedding(x, valid_len)
        output_dict = {'embedding': x}
        return output_dict
