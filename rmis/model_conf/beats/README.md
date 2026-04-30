# BEATs

## INFO

- Official website: https://github.com/microsoft/unilm/tree/master/beats

## Setup

We evaluate the [pre-trained iter3 checkpoint](https://1drv.ms/u/s!AqeByhGUtINrgcpxJUNDxg4eU0r-vA?e=qezPJ5) on RMIS. We find that the pre-trained iter3 checkpoint is more robust for transfer evaluation than other checkpoints.

After downloading the checkpoint, you need to modify the `ckpt` key in `rmis/model_conf/beats/iter3.yaml` as the local checkpoint path. You can use either absolute path or relative path to the `model_dir` stated in `conf/basic.yaml`.
