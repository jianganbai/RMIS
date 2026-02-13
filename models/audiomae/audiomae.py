import torch
import logging
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, Any

from .models_vit import vit_base_patch16


class AudioMAE(nn.Module):
    def __init__(
        self,
        ckpt: str,
        MEAN: float = -4.2677393,
        STD: float = 4.5689974,
    ) -> None:
        super().__init__()
        self.MEAN = MEAN
        self.STD = STD
        self.model = vit_base_patch16(
            img_size=(1024, 128),
            in_chans=1,
            num_classes=0,
            global_pool=True
        )
        checkpoint = torch.load(ckpt, map_location='cpu', weights_only=False)
        checkpoint_model = checkpoint['model']
        state_dict = self.model.state_dict()

        # remove decoder weights
        del_list = []
        for k in checkpoint_model.keys():
            if k.startswith('decoder'):
                del_list.append(k)
        for k in del_list:
            del checkpoint_model[k]

        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        msg = self.model.load_state_dict(checkpoint_model, strict=False)
        logging.info(msg)

    def embedding(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        # AudioMAE only accepts 1024-frame input
        if x.shape[1] < 1024:
            x = F.pad(x, (0, 0, 0, 1024 - x.shape[1]))
        else:
            x = x[:, :1024]
        x = (x - self.MEAN) / (self.STD * 2)  # normalize
        x = x.view(-1, 1, 1024, 128)  # add batch dim and channel dim

        x = self.model(x)  # [bs, emb_size]
        return x

    def forward(
        self,
        x: torch.Tensor,
        **kwargs
    ) -> Dict[str, Any]:
        x = self.embedding(x)
        output_dict = {'embedding': x}
        return output_dict
