# FISHER

## INFO

- Official GitHub website: https://github.com/jianganbai/FISHER

- Official Huggingface: https://huggingface.co/collections/jiangab/fisher

## Setup

FISHER is released on both [GitHub](https://github.com/jianganbai/FISHER) and [Hugging Face](https://huggingface.co/collections/jiangab/fisher). In the RMIS codebase, we implement FISHER based on the GitHub version, which requires users to manually download the checkpoints in advance. You can download the checkpoints from the [official GitHub website](https://github.com/jianganbai/FISHER).

After downloading the checkpoints, you need to modify the `ckpt` key in `rmis/model_conf/fisher/your-interested-scale.yaml` as the local checkpoint path. You can use either absolute path or relative path to the `model_dir` stated in `conf/basic.yaml`.
## Evaluation

For example, evaluate FISHER-small with:

```shell
python -m rmis.scripts.reg_all \
    --model_conf rmis/model_conf/fisher/small.yaml \
    --rel_exp_dir fisher_small \
    --gpu 0
```

Replace `small.yaml` with `tiny.yaml` or `mini.yaml` to evaluate another variant. For multi-GPU evaluation, replace `0` with a comma-separated list of available GPUs.
