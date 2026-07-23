# PaSST

## INFO

- Official website: https://github.com/kkoutini/PaSST
- Input: audio waveform
- Variant: PaSST-S-476
- Representation: pretrained general-audio embedding

## Setup

We evaluate **PaSST-S-476** on RMIS. Download the checkpoint from the [official repository](https://github.com/kkoutini/PaSST), then set `ckpt` in `rmis/model_conf/passt/s-476.yaml` to the local checkpoint path. The path may be absolute or relative to the `model_dir` in `conf/basic.yaml`.

Run the RMIS evaluation with:

```shell
python -m rmis.scripts.reg_all \
    --model_conf rmis/model_conf/passt/s-476.yaml \
    --rel_exp_dir passt \
    --gpu 0
```

For multi-GPU evaluation, replace `0` with a comma-separated list of available GPUs.
