# CoWS

## INFO

- Paper: CoWS: Self-Supervised Representation Pre-Training for Cross-Machine Fault Diagnosis
- Input: vibration waveform
- Representation: self-supervised cross-machine signal embedding

## Setup

We evaluate the pretrained **CoWS** encoder on RMIS using the checkpoint trained on bearing data under load condition 0. Set `ckpt` in `rmis/model_conf/cows/reg/wav_encoder.yaml` to the local checkpoint path. The path may be absolute or relative to the `model_dir` in `conf/basic.yaml`.

Run the RMIS evaluation with:

```shell
python -m rmis.scripts.reg_all \
    --model_conf rmis/model_conf/cows/reg/wav_encoder.yaml \
    --rel_exp_dir cows \
    --gpu 0
```

For multi-GPU evaluation, replace `0` with a comma-separated list of available GPUs.
