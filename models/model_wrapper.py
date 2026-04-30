import os
import torch.nn as nn

from typing import Dict, Any


def construct_cls_model(args: Dict[str, Any]) -> nn.Module:
    model_conf = args.get('model_conf', {})
    for key in ['ckpt', 'weight_ckpt']:
        if key in model_conf.keys() and not os.path.exists(model_conf[key]):
            for dirn in ['model_dir']:
                full_path = os.path.join(args[dirn], model_conf[key])
                if os.path.exists(full_path):
                    model_conf[key] = full_path

    if args['model_name'] == 'audiomae':
        from models.audiomae.audiomae import AudioMAE
        net = AudioMAE(**model_conf)

    elif args['model_name'] == 'beats':
        from models.beats.beats_ft import BEATs_FT
        net = BEATs_FT(**model_conf)

    elif args['model_name'] == 'eat':
        from models.eat.EAT import EAT
        net = EAT(**model_conf)

    elif args['model_name'] == 'ced':
        from models.ced.ced import CED
        net = CED(**model_conf)

    elif args['model_name'] == 'dasheng':
        from models.dasheng.dasheng_ft import Dasheng
        net = Dasheng(**model_conf)

    elif args['model_name'] == 'fisher':
        from models.fisher.fisher import FISHER_infer
        net = FISHER_infer(**model_conf)

    else:
        raise KeyError(f"Unknown model {args['model_name']}!")

    return net
