# Audio Flamingo 3

## INFO

- Official Hugging Face website: https://huggingface.co/nvidia/audio-flamingo-3
- Input: audio waveform
- Representation: pretrained audio encoder embedding

## Setup

We evaluate the audio encoder of **Audio Flamingo 3** on RMIS. The language-model decoder is not required for representation extraction.

Prepare the audio encoder checkpoint and set `ckpt` in `rmis/model_conf/audioflamingo3/encoder.yaml`. The path may be absolute or relative to the `model_dir` in `conf/basic.yaml`.

Run the RMIS evaluation with:

```shell
python -m rmis.scripts.reg_all \
    --model_conf rmis/model_conf/audioflamingo3/encoder.yaml \
    --rel_exp_dir audioflamingo3 \
    --gpu 0
```

For multi-GPU evaluation, replace `0` with a comma-separated list of available GPUs.
