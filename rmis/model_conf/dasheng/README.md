# DaSheng

## INFO

- Official website: https://github.com/XiaoMi/dasheng

## Setup

We evaluate **DaSheng-base**, **DaSheng-0.6B** and **DaSheng-1.2B** on RMIS. We borrow the code from the [official website](https://github.com/XiaoMi/dasheng), which supports automatic checkpoint downloading from zenodo on the first run. If it is the first run, run the following command to first download the checkpoint with a single process.

```python
python -m rmis.scripts.reg_all \
    --model_conf rmis/model_conf/dasheng/{scale}.yaml \
    --rel_exp_dir dummy \
    --gpu 0
```

After downloading, you can terminate the process and run the evaluation command with multiple GPUs.
