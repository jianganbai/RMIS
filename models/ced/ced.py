import torch
import logging
import torch.nn as nn

from typing import Dict, Any

from .models.audiotransformer import AudioTransformer


class CED(nn.Module):
    def __init__(
        self,
        scale: str,
        ckpt: str,
    ) -> None:
        super().__init__()
        if scale == 'base':
            model_kwargs = dict(
                patch_size=16,
                embed_dim=768,
                depth=12,
                num_heads=12,
                mlp_ratio=4,
                outputdim=527  # dummy. model output last layer embed
            )
        elif scale == 'small':
            model_kwargs = dict(
                patch_size=16,
                embed_dim=384,
                depth=12,
                num_heads=6,
                mlp_ratio=4,
                outputdim=527  # dummy. model output last layer embed
            )
        elif scale == 'mini':
            model_kwargs = dict(
                patch_size=16,
                embed_dim=256,
                depth=12,
                num_heads=4,
                mlp_ratio=4,
                outputdim=527  # dummy. model output last layer embed
            )
        elif scale == 'tiny':
            model_kwargs = dict(
                patch_size=16,
                embed_dim=192,
                depth=12,
                num_heads=3,
                mlp_ratio=4,
                outputdim=527  # dummy. model output last layer embed
            )
        else:
            raise NotImplementedError(f'Scale {scale} has not been implemented!')
        self.model = AudioTransformer(**model_kwargs)

        checkpoint = torch.load(ckpt, weights_only=True)
        missing_unexpected = self.model.load_state_dict(checkpoint, strict=False)
        logging.info(missing_unexpected)

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        # No need to normalize. The model has an initial BN that normalizes across fbanks.
        x = self.model(x)
        x = torch.mean(x, dim=1)
        return x.squeeze(1)

    def forward(
        self,
        x: torch.Tensor,
        out_emb: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        x = self.embedding(x)
        output_dict = {'embedding': x}
        return output_dict
