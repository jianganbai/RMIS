# LiConvFormer

## INFO

- Official website: https://github.com/yanshen0210/LiConvFormer-a-lightweight-fault-diagnosis-framework
- Paper: LiConvFormer: A Lightweight Fault Diagnosis Framework Using Separable Multiscale Convolution and Broadcast Self-Attention
- Input: vibration waveform
- Checkpoint: CWRU-trained LiConvFormer

## Setup

We evaluate **LiConvFormer** on RMIS using the CWRU-trained checkpoint. Download the model resources from the [official repository](https://github.com/yanshen0210/LiConvFormer-a-lightweight-fault-diagnosis-framework), then set `ckpt` in `rmis/model_conf/liconvformer/cwru.yaml` to the local checkpoint path. The path may be absolute or relative to the `model_dir` in `conf/basic.yaml`.

Run the RMIS evaluation with:

```shell
python -m rmis.scripts.reg_all \
    --model_conf rmis/model_conf/liconvformer/cwru.yaml \
    --rel_exp_dir liconvformer \
    --gpu 0
```

For multi-GPU evaluation, replace `0` with a comma-separated list of available GPUs.
