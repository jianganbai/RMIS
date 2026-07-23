# Time-MoE

## INFO

- Official website: https://github.com/Time-MoE/Time-MoE
- Official Hugging Face website: https://huggingface.co/Maple728/TimeMoE-50M
- Input: time series waveform
- Variant: TimeMoE-50M

## Setup

We evaluate **TimeMoE-50M** on RMIS. Set `ckpt` in `rmis/model_conf/time_moe/base.yaml` to the local Time-MoE model directory. The path may be absolute or relative to the `model_dir` in `conf/basic.yaml`.

Run the RMIS evaluation with:

```shell
python -m rmis.scripts.reg_all \
    --model_conf rmis/model_conf/time_moe/base.yaml \
    --rel_exp_dir time_moe \
    --gpu 0
```

For multi-GPU evaluation, replace `0` with a comma-separated list of available GPUs.
