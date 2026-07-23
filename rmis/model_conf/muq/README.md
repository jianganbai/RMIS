# MuQ

## INFO

- Official Hugging Face website: https://huggingface.co/OpenMuQ/MuQ-large-msd-iter
- Input: music waveform
- Variant: MuQ-large-msd-iter
- Representation: self-supervised music audio embedding

## Setup

We evaluate **MuQ-large-msd-iter** on RMIS. The checkpoint is downloaded automatically from Hugging Face on the first run. To use a local copy, set `model_id` in `rmis/model_conf/muq/msd.yaml` to the local model directory.

Run the RMIS evaluation with:

```shell
python -m rmis.scripts.reg_all \
    --model_conf rmis/model_conf/muq/msd.yaml \
    --rel_exp_dir muq \
    --gpu 0
```

For multi-GPU evaluation, replace `0` with a comma-separated list of available GPUs.
