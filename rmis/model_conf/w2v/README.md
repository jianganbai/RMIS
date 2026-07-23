# Wav2Vec 2.0

## INFO

- Official Hugging Face website: https://huggingface.co/facebook
- Input: speech or audio waveform
- Representation: self-supervised speech embedding
- Variants: base, large, 1B, and 2B

## Setup

We provide four Wav2Vec 2.0 configurations:

- `base.yaml`: `facebook/wav2vec2-base-960h`
- `large.yaml`: `facebook/wav2vec2-xls-r-300m`
- `1b.yaml`: `facebook/wav2vec2-xls-r-1b`
- `2b.yaml`: `facebook/wav2vec2-xls-r-2b`

The checkpoints are downloaded automatically from Hugging Face on the first run.

For example, evaluate the base model with:

```shell
python -m rmis.scripts.reg_all \
    --model_conf rmis/model_conf/w2v/base.yaml \
    --rel_exp_dir w2v_base \
    --gpu 0
```

Replace `base.yaml` with `large.yaml`, `1b.yaml`, or `2b.yaml` to evaluate another variant. For multi-GPU evaluation, replace `0` with a comma-separated list of available GPUs.
