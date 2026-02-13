# FISHER

## INFO

- Official GitHub website: https://github.com/jianganbai/FISHER

- Official Huggingface: https://huggingface.co/collections/jiangab/fisher

## Setup

FISHER is released on both [GitHub](https://github.com/jianganbai/FISHER) and [Hugging Face](https://huggingface.co/collections/jiangab/fisher). In the RMIS codebase, we implement FISHER based on the GitHub version, which requires users to manually download the checkpoints in advance. You can download the checkpoints from the [official GitHub website](https://github.com/jianganbai/FISHER).

After downloading the checkpoints, you need to modify the `ckpt` key in `rmis/model_conf/fisher/your-interested-scale.yaml` as the local checkpoint path. You can use either absolute path or relative path to the `model_dir` stated in `conf/basic.yaml`.
