# RotLLM

## INFO

- Official website: https://github.com/SIA-IDE/RotLLM
- Input: vibration waveform
- Representation: pretrained rotating-machinery signal embedding

## Setup

We evaluate the pretrained **RotLLM** signal encoder on RMIS. Download the model resources from the [official repository](https://github.com/SIA-IDE/RotLLM), then set `ckpt` in `rmis/model_conf/rotllm/sfn.yaml` to the local encoder checkpoint path. The path may be absolute or relative to the `model_dir` in `conf/basic.yaml`.

Run the RMIS evaluation with:

```shell
python -m rmis.scripts.reg_all \
    --model_conf rmis/model_conf/rotllm/sfn.yaml \
    --rel_exp_dir rotllm \
    --gpu 0
```

For multi-GPU evaluation, replace `0` with a comma-separated list of available GPUs.
