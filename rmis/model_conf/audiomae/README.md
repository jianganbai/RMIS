# AudioMAE

## INFO

- Official website: https://github.com/facebookresearch/AudioMAE

## Setup

Since only the base version is open-sourced, we evaluate the [pre-trained base checkpoint](https://drive.google.com/file/d/1ni_DV4dRf7GxM8k-Eirx71WP9Gg89wwu/view?usp=share_link) on RMIS.

After downloading the checkpoint, you need to modify the `ckpt` key in `rmis/model_conf/audiomae/base.yaml` as the local checkpoint path. You can use either absolute path or relative path to the `model_dir` stated in `conf/basic.yaml`.
## Evaluation

For example, evaluate AudioMAE-base with:

```shell
python -m rmis.scripts.reg_all \
    --model_conf rmis/model_conf/audiomae/base.yaml \
    --rel_exp_dir audiomae \
    --gpu 0
```

For multi-GPU evaluation, replace `0` with a comma-separated list of available GPUs.
