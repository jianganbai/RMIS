# PEEMD

## INFO

- Paper: Pseudo-extrema-based EMD and Its Application to Rotor Fault Diagnosis
- Input: industrial signal waveform
- Type: non-parametric signal decomposition baseline
- Checkpoint: not required

## Setup

PEEMD extracts signal representations through empirical mode decomposition and permutation-entropy-based features. No pretrained checkpoint is required. Use `rmis/model_conf/peemd/base.yaml` for evaluation. The `num_proc` option controls the number of processes used for feature extraction.

Run the RMIS evaluation with:

```shell
python -m rmis.scripts.reg_all \
    --model_conf rmis/model_conf/peemd/base.yaml \
    --rel_exp_dir peemd \
    --gpu 0
```

For multi-GPU evaluation, replace `0` with a comma-separated list of available GPUs.
