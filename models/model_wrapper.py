import os
import torch.nn as nn

from typing import Dict, Any


def construct_cls_model(args: Dict[str, Any]) -> nn.Module:
    # Resolve local checkpoint paths on a copy so wrapper construction does not
    # mutate the caller-provided config dictionary in place.
    model_conf = dict(args.get('model_conf', {}))
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

    elif args['model_name'] == 'w2v':
        from models.w2v.w2v import W2V
        net = W2V(**model_conf)

    elif args['model_name'] == 'mert':
        from models.mert.mert import MERT
        net = MERT(**model_conf)

    elif args['model_name'] == 'passt':
        from models.passt.passt import PaSST
        net = PaSST(**model_conf)

    elif args['model_name'] == 'whisper':
        from models.whisper.whisper import Whisper
        net = Whisper(**model_conf)

    elif args['model_name'] in ['qwen2_audio', 'audioflamingo3']:
        from models.qwen2_audio.qwen2_audio import Qwen2_Audio
        net = Qwen2_Audio(**model_conf)

    elif args['model_name'] == 'qwen_audio2_5':
        from models.qwen2_5_omni.qwen2_5_omni import Qwen2_5_Audio
        net = Qwen2_5_Audio(**model_conf)

    elif args['model_name'] == 'tfpred':
        from models.tfpred.tfpred import TFPred
        net = TFPred(**model_conf)

    elif args['model_name'] == 'liconvformer':
        from models.liconvformer.liconvformer import LiConvFormer_FT
        net = LiConvFormer_FT(**model_conf)
    elif args['model_name'] == 'bearllm':
        from models.bearllm.bearllm import BearLLMFCN
        net = BearLLMFCN(**model_conf)

    elif args['model_name'] == 'rotllm':
        from models.rotllm.sfn import SFN
        net = SFN(**model_conf)

    elif args['model_name'] == 'peemd':
        from models.peemd.peemd import PEEMDModel
        net = PEEMDModel(loss=None, emb_size=0, **model_conf)

    elif args['model_name'] == 'cows':
        from models.cows.cows import CoWS
        net = CoWS(**model_conf)

    elif args['model_name'] == 'echo':
        from models.echo.echo import ECHO
        net = ECHO(**model_conf)

    elif args['model_name'] == 'time_moe':
        from models.time_moe.time_moe import TimeMoE
        net = TimeMoE(**model_conf)

    elif args['model_name'] == 'sundial':
        from models.sundial.sundial import Sundial
        net = Sundial(**model_conf)

    elif args['model_name'] == 'muq':
        from models.muq.muq import MuQMSD
        net = MuQMSD(**model_conf)

    else:
        raise KeyError(f"Unknown model {args['model_name']}!")

    return net
