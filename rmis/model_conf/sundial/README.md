# Sundial

## INFO

- Official website: https://github.com/thuml/Sundial
- Official Hugging Face website: https://huggingface.co/thuml/sundial-base-128m
- Input: time series waveform
- Variant: Sundial-base-128m

## Setup

We evaluate **Sundial-base-128m** on RMIS. The checkpoint is downloaded automatically from Hugging Face on the first run. To use a local copy, set `ckpt` in `rmis/model_conf/sundial/base.yaml` to the local model directory.

Run the RMIS evaluation with:

```shell
python -m rmis.scripts.reg_all \
    --model_conf rmis/model_conf/sundial/base.yaml \
    --rel_exp_dir sundial \
    --gpu 0
```

For multi-GPU evaluation, replace `0` with a comma-separated list of available GPUs.
