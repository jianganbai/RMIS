import logging
import torch
import torch.nn as nn

from typing import Dict, Any, Literal, Optional

from .bearllm_fcn import SingleInputEncoder


# RMIS adapter for BearLLM FCN features. The imported FCN module contains
# the model implementation while this file handles RMIS pooling/output glue.
class BearLLMFCN(nn.Module):
    def __init__(
        self,
        ckpt: str,
        feat_aggre: Literal['raw', 'mean_pool'] = 'mean_pool',
        chunk_size: int = 24000,
        emb_size: int = 0,
    ) -> None:
        super().__init__()
        if feat_aggre not in ['raw', 'mean_pool']:
            raise ValueError(f"Unsupported feat_aggre: {feat_aggre}")

        self.backbone = SingleInputEncoder()
        self.feat_aggre = feat_aggre
        self.chunk_size = int(chunk_size)

        full_state = torch.load(ckpt, map_location='cpu', weights_only=True)
        fe_state = {
            key.replace('feature_encoder.', ''): value
            for key, value in full_state.items()
            if key.startswith('feature_encoder.')
        }
        missing_unexpected = self.backbone.feature_encoder.load_state_dict(fe_state, strict=True)
        logging.info(f"Loaded BearLLM FCN checkpoint: {missing_unexpected}")

        self.output_dim = 128
        if emb_size > 0 and emb_size != self.output_dim:
            self.fc = nn.Linear(self.output_dim, emb_size)

    def _chunk(self, x: torch.Tensor):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.dim() != 3 or x.size(1) != 1:
            raise ValueError(f"Expected x as [B, L] or [B, 1, L], got {tuple(x.shape)}")

        num_chunks = (x.size(-1) + self.chunk_size - 1) // self.chunk_size
        chunks = []
        for idx in range(num_chunks):
            start = idx * self.chunk_size
            end = start + self.chunk_size
            chunk = x[:, :, start:end]
            if chunk.size(-1) < self.chunk_size:
                chunk = torch.nn.functional.pad(chunk, (0, self.chunk_size - chunk.size(-1)))
            chunks.append(chunk)
        return chunks

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        feats = [self.backbone(chunk) for chunk in self._chunk(x)]
        x = torch.stack(feats, dim=1).mean(dim=1)

        if self.feat_aggre == 'raw':
            return x

        x = x.mean(dim=-1)
        if hasattr(self, 'fc'):
            x = self.fc(x)
        return x

    def forward(
        self,
        x: torch.Tensor,
        **kwargs,
    ) -> Dict[str, Any]:
        x = self.embedding(x)
        output_dict = {'embedding': x}
        return output_dict
