# EAT

## INFO

- Official GitHub website: https://github.com/cwx-worst-one/EAT

- Official Hugging Face 🤗 website: https://huggingface.co/collections/worstchan/eat


## Setup


We evaluate **EAT-base30** and **EAT-large** on RMIS. You can find the introduction to the EAT model on the [official GitHub website](https://github.com/cwx-worst-one/EAT). However, the official GitHub implementation requires [fairseq](https://github.com/facebookresearch/fairseq) to be installed, which is fairly cumbersome. Therefore, we implement EAT by the [offical Hugging Face 🤗](https://huggingface.co/collections/worstchan/eat) version. Model checkpoints will be automatically downloaded from huggingface on the first run. If it is the first run, run the following command to first download the checkpoint with a single process.

```python
python -m rmis.scripts.reg_all \
    --model_conf rmis/model_conf/dasheng/{scale}.yaml \
    --rel_exp_dir dummy \
    --gpu 0
```

After downloading, you can terminate the process and run the evaluation command with multiple GPUs.
