# ECHO

## INFO

- Official website: https://github.com/yucongzh/ECHO
- Paper: ECHO: Frequency-aware Hierarchical Encoding for Variable-length Signal
- Input: audio, vibration, and other machine-signal waveforms
- Variants: ECHO-tiny and ECHO-small

## Setup

We evaluate **ECHO-tiny** and **ECHO-small** on RMIS. Download the checkpoints from the [official repository](https://github.com/yucongzh/ECHO), then set `ckpt` in the corresponding configuration under `rmis/model_conf/echo/`. The path may be absolute or relative to the `model_dir` in `conf/basic.yaml`.

For example, evaluate ECHO-small with:

```shell
python -m rmis.scripts.reg_all \
    --model_conf rmis/model_conf/echo/small.yaml \
    --rel_exp_dir echo_small \
    --gpu 0
```

Replace `small.yaml` with `tiny.yaml` to evaluate ECHO-tiny. For multi-GPU evaluation, replace `0` with a comma-separated list of available GPUs.
