# TFPred

## INFO

- Official website: https://github.com/Xiaohan-Chen/TFPred
- Paper: TFPred: Learning Discriminative Representations from Unlabeled Data for Few-Label Rotating Machinery Fault Diagnosis
- Input: rotating-machinery waveform
- Representation: pretrained time-frequency prediction embedding

## Setup

We evaluate the pretrained **TFPred** time encoder on RMIS. Download the checkpoint from the [official repository](https://github.com/Xiaohan-Chen/TFPred), then set `ckpt` in `rmis/model_conf/tfpred/reg/time_encoder.yaml` to the local checkpoint path. The path may be absolute or relative to the `model_dir` in `conf/basic.yaml`.

Run the RMIS evaluation with:

```shell
python -m rmis.scripts.reg_all \
    --model_conf rmis/model_conf/tfpred/reg/time_encoder.yaml \
    --rel_exp_dir tfpred \
    --gpu 0
```

For multi-GPU evaluation, replace `0` with a comma-separated list of available GPUs.
