# BearLLM

## INFO

- Official website: https://github.com/SIA-IDE/BearLLM
- Paper: BearLLM: A Prior Knowledge-Enhanced Bearing Health Management Framework with Unified Vibration Signal Representation
- Input: vibration waveform
- Representation: pretrained bearing-signal encoder embedding

## Setup

We evaluate the vibration encoder of **BearLLM** on RMIS. Download the model resources from the [official repository](https://github.com/SIA-IDE/BearLLM), then set `ckpt` in `rmis/model_conf/bearllm/encoder.yaml` to the local checkpoint path. The path may be absolute or relative to the `model_dir` in `conf/basic.yaml`.

Run the RMIS evaluation with:

```shell
python -m rmis.scripts.reg_all \
    --model_conf rmis/model_conf/bearllm/encoder.yaml \
    --rel_exp_dir bearllm \
    --gpu 0
```

For multi-GPU evaluation, replace `0` with a comma-separated list of available GPUs.
