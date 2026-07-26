import torch
import torch.nn as nn

from typing import Dict, Any, Optional
from transformers import AutoModelForCausalLM


# RMIS adapter for Time-MoE style Hugging Face causal backbones.
class TimeMoE(nn.Module):
    def __init__(
        self,
        ckpt: str,
        freeze: bool = True,
        feat_aggre: str = 'mean',
        trust_remote_code: bool = True,
        torch_dtype: Optional[str] = None,
        device_map: Optional[str] = None,
        emb_size: int = 0,
    ) -> None:
        super().__init__()
        if feat_aggre not in ['mean', 'last', 'max']:
            raise ValueError(f"Unsupported feat_aggre: {feat_aggre}")

        dtype = torch.float32
        if torch_dtype == 'bf16':
            dtype = torch.bfloat16
        elif torch_dtype == 'fp16':
            dtype = torch.float16

        self.backbone = AutoModelForCausalLM.from_pretrained(
            ckpt,
            trust_remote_code=trust_remote_code,
            device_map=device_map,
            torch_dtype=dtype,
        )
        self.feat_aggre = feat_aggre

        hidden_size = getattr(self.backbone.config, 'hidden_size', None)
        if hidden_size is None:
            hidden_size = getattr(self.backbone.config, 'd_model', None)
        if hidden_size is None:
            raise RuntimeError("Cannot infer hidden size from backbone config.")

        self.proj = None
        if emb_size > 0:
            self.proj = nn.Identity() if hidden_size == emb_size else nn.Linear(hidden_size, emb_size, bias=False)

        if freeze:
            self.backbone.eval()
            self.backbone.requires_grad_(False)

    @staticmethod
    def _make_attention_mask(valid_len: torch.Tensor, length: int) -> torch.Tensor:
        idx = torch.arange(length, device=valid_len.device).unsqueeze(0)
        return (idx < valid_len.unsqueeze(1)).to(torch.long)

    def _pool(self, hidden_states: torch.Tensor, valid_len: Optional[torch.Tensor]) -> torch.Tensor:
        batch_size, length, _ = hidden_states.shape
        if valid_len is None:
            if self.feat_aggre == 'mean':
                return hidden_states.mean(dim=1)
            if self.feat_aggre == 'max':
                return hidden_states.max(dim=1).values
            return hidden_states[:, -1, :]

        attn = self._make_attention_mask(valid_len, length)
        mask = attn.unsqueeze(-1).to(hidden_states.dtype)

        if self.feat_aggre == 'mean':
            denom = mask.sum(dim=1).clamp_min(1.0)
            return (hidden_states * mask).sum(dim=1) / denom

        if self.feat_aggre == 'max':
            neg_inf = torch.finfo(hidden_states.dtype).min
            return hidden_states.masked_fill(mask.eq(0), neg_inf).max(dim=1).values

        idx = (valid_len.clamp_min(1) - 1).to(torch.long)
        return hidden_states[torch.arange(batch_size, device=hidden_states.device), idx, :]

    def embedding(
        self,
        x: torch.Tensor,
        valid_len: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x.dim() not in [2, 3]:
            raise ValueError(f"Expected x as [B, L] or [B, L, 1], got {tuple(x.shape)}")

        attention_mask = None
        if valid_len is not None:
            attention_mask = self._make_attention_mask(valid_len, x.shape[1])

        with torch.autocast('cuda', enabled=False):
            x = x.float()
            x = x / (x.std(dim=1, keepdim=True) + 1e-6)
            outputs = self.backbone.model(
                input_ids=x,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
            )

        x = self._pool(outputs.last_hidden_state, valid_len)
        if self.proj is not None:
            x = self.proj(x)
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
