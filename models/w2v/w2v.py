import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, Any, Optional
from transformers import AutoFeatureExtractor, AutoModel


# RMIS adapter for wav2vec-style Hugging Face backbones loaded by model_id.
class W2V(nn.Module):
    def __init__(
        self,
        model_id: str,
        cache_dir: Optional[str] = None,
        freeze: bool = True,
        output_norm: bool = False,
        weighted_sum: bool = True,
    ) -> None:
        super().__init__()
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            model_id,
            cache_dir=cache_dir,
        )
        self.model = AutoModel.from_pretrained(
            model_id,
            cache_dir=cache_dir,
        )

        self.freeze = freeze
        self.output_norm = output_norm
        self.normalize_wav = getattr(self.feature_extractor, 'do_normalize', False)
        self.weighted_sum = weighted_sum

        if weighted_sum:
            num_layers = self.model.config.num_hidden_layers + 1
            self.layer_weights = nn.Parameter(torch.zeros(num_layers))

        if freeze:
            self.model.eval()
            self.model.requires_grad_(False)

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize_wav:
            x = F.layer_norm(x, x.shape)

        if self.weighted_sum:
            output = self.model(x, output_hidden_states=True)
            hidden_states = torch.stack(output.hidden_states, dim=0)
            weights = F.softmax(self.layer_weights, dim=0)
            x = torch.sum(hidden_states * weights[:, None, None, None], dim=0)
        else:
            x = self.model(x).last_hidden_state

        if self.output_norm:
            x = F.layer_norm(x, x.shape[1:])
        return x

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        if self.freeze:
            with torch.no_grad():
                x = self._extract_features(x)
        else:
            x = self._extract_features(x)
        return torch.mean(x, dim=1)

    def forward(
        self,
        x: torch.Tensor,
        **kwargs
    ) -> Dict[str, Any]:
        x = self.embedding(x)
        output_dict = {'embedding': x}
        return output_dict
