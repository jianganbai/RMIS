# Whisper

## INFO

- Official website: https://github.com/openai/whisper
- Input: audio waveform
- Representation: pretrained speech/audio encoder embedding
- Variants: tiny, base, small, medium, and large-v3

## Setup

We provide five Whisper encoder configurations:

- `tiny.yaml`
- `base.yaml`
- `small.yaml`
- `medium.yaml`
- `large-v3.yaml`

After preparing the encoder checkpoint, set `ckpt` in the corresponding yaml file under `rmis/model_conf/whisper/` to the local checkpoint path. The path may be absolute or relative to the `model_dir` in `conf/basic.yaml`.

For example, evaluate Whisper-base with:

```shell
python -m rmis.scripts.reg_all \
    --model_conf rmis/model_conf/whisper/base.yaml \
    --rel_exp_dir whisper_base \
    --gpu 0
```

Replace `base.yaml` with another variant to evaluate a different checkpoint. For multi-GPU evaluation, replace `0` with a comma-separated list of available GPUs.
