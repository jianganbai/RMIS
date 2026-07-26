# MiDaShengLM

## INFO

- Official website: https://github.com/Audio-WestlakeU/MiDaShengLM
- Input: audio waveform
- Variant: MiDaShengLM-0.6B audio encoder
- Representation: pretrained general-audio embedding

## Setup

We evaluate the **MiDaShengLM-0.6B** audio encoder on RMIS. Download the model resources from the [official repository](https://github.com/Audio-WestlakeU/MiDaShengLM), then set `weight_ckpt` in `rmis/model_conf/midashenglm/0.6b_encoder.yaml` to the local audio-encoder checkpoint. The path may be absolute or relative to the `model_dir` in `conf/basic.yaml`.

Run the RMIS evaluation with:

```shell
python -m rmis.scripts.reg_all \
    --model_conf rmis/model_conf/midashenglm/0.6b_encoder.yaml \
    --rel_exp_dir midashenglm \
    --gpu 0
```

For multi-GPU evaluation, replace `0` with a comma-separated list of available GPUs.
