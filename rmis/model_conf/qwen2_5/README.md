# Qwen2.5-Omni

## INFO

- Official Hugging Face website: https://huggingface.co/Qwen/Qwen2.5-Omni-7B
- Input: audio waveform
- Representation: pretrained audio encoder embedding

## Setup

We evaluate the audio encoder of **Qwen2.5-Omni** on RMIS. Prepare the audio encoder checkpoint and set `ckpt` in `rmis/model_conf/qwen2_5/omni_encoder.yaml` to the local checkpoint path. The path may be absolute or relative to the `model_dir` in `conf/basic.yaml`.

Run the RMIS evaluation with:

```shell
python -m rmis.scripts.reg_all \
    --model_conf rmis/model_conf/qwen2_5/omni_encoder.yaml \
    --rel_exp_dir qwen2_5_omni \
    --gpu 0
```

For multi-GPU evaluation, replace `0` with a comma-separated list of available GPUs.
