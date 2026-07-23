import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, Any, Literal, Optional
from transformers import AutoModel


# RMIS adapter for MERT checkpoints hosted on Hugging Face.
class MERT(nn.Module):
    def __init__(
        self,
        ver: Literal['95M', '330M'],
        weighted_sum: bool = False,
        feat_aggre: Literal['mean_pool'] = 'mean_pool',
        freeze: bool = True,
    ) -> None:
        super().__init__()
        if ver not in ['95M', '330M']:
            raise ValueError(f'Unsupported MERT version {ver}')
        if feat_aggre != 'mean_pool':
            raise ValueError(f'Unsupported feat_aggre: {feat_aggre}')

        self.model = AutoModel.from_pretrained(
            f'm-a-p/MERT-v1-{ver}',
            trust_remote_code=True,
        )
        self.weighted_sum = weighted_sum
        self.freeze = freeze

        if weighted_sum:
            num_layers = self.model.config.num_hidden_layers + 1
            self.feat_weight = nn.Parameter(torch.zeros(num_layers, dtype=torch.float32))

        if freeze:
            self.model.eval()
            self.model.requires_grad_(False)

    @staticmethod
    def _make_attention_mask(valid_len: torch.Tensor, length: int) -> torch.Tensor:
        idx = torch.arange(length, device=valid_len.device).unsqueeze(0)
        return (idx < valid_len.unsqueeze(1)).to(torch.long)

    def _extract_features(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.weighted_sum:
            outputs = self.model(
                input_values=x,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden_states = torch.stack(outputs.hidden_states, dim=0)
            weights = F.softmax(self.feat_weight, dim=0)
            x = torch.sum(hidden_states * weights[:, None, None, None], dim=0)
        else:
            outputs = self.model(input_values=x, attention_mask=attention_mask)
            x = outputs.last_hidden_state
        return x

    def embedding(
        self,
        x: torch.Tensor,
        valid_len: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attention_mask = None
        if valid_len is not None:
            valid_len = valid_len.to(device=x.device, dtype=torch.long).clamp_(0, x.shape[1])
            attention_mask = self._make_attention_mask(valid_len, x.shape[1])

        if self.freeze:
            with torch.no_grad():
                x = self._extract_features(x, attention_mask=attention_mask)
        else:
            x = self._extract_features(x, attention_mask=attention_mask)

        if attention_mask is None:
            return torch.mean(x, dim=1)

        mask = attention_mask.unsqueeze(-1).to(x.dtype)
        return (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def forward(
        self,
        x: torch.Tensor,
        valid_len: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        x = self.embedding(x, valid_len)
        output_dict = {'embedding': x}
        return output_dict
