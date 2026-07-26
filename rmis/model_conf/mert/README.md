# MERT

## INFO

- Official Hugging Face website: https://huggingface.co/m-a-p
- Input: music waveform
- Variants: MERT-v1-95M and MERT-v1-330M
- Representation: self-supervised music audio embedding

## Setup

We evaluate **MERT-v1-95M** and **MERT-v1-330M** on RMIS. The corresponding configurations are `v1-95m.yaml` and `v1-330m.yaml`. The checkpoints are downloaded automatically from Hugging Face on the first run.

For example, evaluate MERT-v1-95M with:

```shell
python -m rmis.scripts.reg_all \
    --model_conf rmis/model_conf/mert/v1-95m.yaml \
    --rel_exp_dir mert_95m \
    --gpu 0
```

Replace `v1-95m.yaml` with `v1-330m.yaml` to evaluate MERT-v1-330M. For multi-GPU evaluation, replace `0` with a comma-separated list of available GPUs.
