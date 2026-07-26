# CED

## INFO

- Official website: https://github.com/RicherMans/CED

## Setup

We evaluate **CED-tiny**, **CED-mini**, **CED-small** and **CED-base** on RMIS. You can download them from the [official website](https://github.com/RicherMans/CED). These checkpoints can be evaluated independently, so you can only download the checkpoint that you are interested in.

After downloading the checkpoints, you need to modify the `ckpt` key in `rmis/model_conf/ced/your-interested-scale.yaml` as the local checkpoint path. You can use either absolute path or relative path to the `model_dir` stated in `conf/basic.yaml`.
## Evaluation

For example, evaluate CED-base with:

```shell
python -m rmis.scripts.reg_all \
    --model_conf rmis/model_conf/ced/base.yaml \
    --rel_exp_dir ced_base \
    --gpu 0
```

Replace `base.yaml` with `tiny.yaml`, `mini.yaml`, or `small.yaml` to evaluate another variant. For multi-GPU evaluation, replace `0` with a comma-separated list of available GPUs.
