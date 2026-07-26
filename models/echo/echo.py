import torch
import torch.nn as nn

from typing import Dict, Any, Optional

from .audioMAE_band_upgrade import AudioMAEWithBand


class ECHO(nn.Module):
    def __init__(
        self,
        ckpt: str,
        sample_rate: int,
        model_size: str = 'small',
        weight_ckpt: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate

        if model_size == 'small':
            model_cfg = {
                'spec_len': 2000,
                'shift_size': 16,
                'in_chans': 1,
                'embed_dim': 384,
                'encoder_depth': 12,
                'num_heads': 6,
                'mlp_ratio': 4.0,
                'norm_layer': lambda x: nn.LayerNorm(x, eps=1e-6),
                'fix_pos_emb': True,
                'band_width': 32,
                'mask_ratio': 0.75,
                'freq_pos_emb_dim': 384,
            }
        elif model_size == 'tiny':
            model_cfg = {
                'spec_len': 2000,
                'shift_size': 16,
                'in_chans': 1,
                'embed_dim': 192,
                'encoder_depth': 12,
                'num_heads': 3,
                'mlp_ratio': 4.0,
                'norm_layer': lambda x: nn.LayerNorm(x, eps=1e-6),
                'fix_pos_emb': True,
                'band_width': 32,
                'mask_ratio': 0.75,
                'freq_pos_emb_dim': 192,
            }
        else:
            raise ValueError(f"Unsupported model_size: {model_size}")

        self.model = AudioMAEWithBand(**model_cfg)
        checkpoint = torch.load(ckpt, map_location='cpu', weights_only=False)
        if 'encoder' in checkpoint:
            self.model.load_state_dict(checkpoint['encoder'])
        else:
            self.model.load_state_dict(checkpoint)

        if weight_ckpt is not None:
            weights = torch.load(weight_ckpt, weights_only=True, map_location='cpu')
            self.load_state_dict(weights, strict=False)

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        spectrograms = self.model.preprocess_batch_audio_to_spectrogram(x, sample_rate=self.sample_rate)
        features = []
        for spec in spectrograms:
            utt_feat, _ = self.model.extract_features(spec, sample_rate=self.sample_rate)
            features.append(utt_feat)
        return torch.stack(features, dim=0).to(x.device)

    def forward(
        self,
        x: torch.Tensor,
        **kwargs
    ) -> Dict[str, Any]:
        self.model.eval()
        with torch.no_grad():
            x = self.embedding(x)
        output_dict = {'embedding': x}
        return output_dict

    def load_state_dict(self, state_dict, strict: bool = False, assign: bool = False):
        key = 'model' if 'model' in state_dict else None
        state_dict = state_dict[key] if key else state_dict
        return super().load_state_dict(state_dict, strict=strict, assign=assign)
