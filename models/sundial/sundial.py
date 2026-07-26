import torch
import torch.nn as nn

from typing import Dict, Any, Optional
from transformers import AutoModelForCausalLM

from .HF.configuration_sundial import SundialConfig
from .HF.modeling_susndial import SundialModel


# RMIS adapter for Sundial. The local HF-style package keeps the imported
# configuration and model implementation separate from RMIS glue code.
class Sundial(nn.Module):
    def __init__(
        self,
        ckpt: str = 'thuml/sundial-base-128m',
        freeze: bool = True,
        emb_size: int = 0,
    ) -> None:
        super().__init__()
        config = SundialConfig.from_pretrained(ckpt)
        pretrained = AutoModelForCausalLM.from_pretrained(
            ckpt,
            trust_remote_code=True,
        )
        self.model = SundialModel(config)
        self.model.load_state_dict(pretrained.model.state_dict(), strict=False)

        if emb_size > 0 and emb_size != 768:
            self.fc = nn.Linear(768, emb_size)

        if freeze:
            self.model.eval()
            self.model.requires_grad_(False)

    @staticmethod
    def _gen_attn_mask(valid_len: Optional[torch.Tensor], seq_len: int) -> Optional[torch.Tensor]:
        if valid_len is None:
            return None
        patch_valid = (valid_len + 15) // 16
        patch_total = (seq_len + 15) // 16
        return torch.arange(patch_total, device=valid_len.device).unsqueeze(0) < patch_valid.unsqueeze(1)

    def embedding(
        self,
        x: torch.Tensor,
        valid_len: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attention_mask = self._gen_attn_mask(valid_len, x.shape[1])
        outputs = self.model(input_ids=x, attention_mask=attention_mask, return_dict=True)
        hidden_states = outputs.last_hidden_state

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
            x = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        else:
            x = hidden_states.mean(dim=1)

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
