# BEATs

## INFO

- BEATs official website: https://github.com/microsoft/unilm/tree/master/beats
- OpenBEATs official website: https://github.com/Audio-WestlakeU/OpenBEATs

## Setup

We provide configs for BEATs iter3 and OpenBEATs:

- `iter3.yaml`
- `open_base.yaml`
- `open_large.yaml`

After downloading the checkpoint, modify the `ckpt` key in the corresponding yaml as the local checkpoint path. You can use either an absolute path or a path relative to the `model_dir` stated in `conf/basic.yaml`.
## Evaluation

For example, evaluate BEATs-iter3 with:

```shell
python -m rmis.scripts.reg_all \
    --model_conf rmis/model_conf/beats/iter3.yaml \
    --rel_exp_dir beats_iter3 \
    --gpu 0
```

Use `open_base.yaml` or `open_large.yaml` to evaluate an OpenBEATs checkpoint. For multi-GPU evaluation, replace `0` with a comma-separated list of available GPUs.
