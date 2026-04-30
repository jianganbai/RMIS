MaFaulDa preprocessing extracts the CSV archives and generates wav datasets in `datasets_wav/mafaulda_sound` and `datasets_wav/mafaulda_vib`.

Before running `mafaulda_preprocess.sh`, the expected structure is:

```
datasets_raw/
└── mafaulda/
    └── full.zip
```

After running `mafaulda_preprocess.sh`, the expected structure is:

```
datasets_raw/
└── mafaulda/
    ├── horizontal-misalignment/
    ├── imbalance/
    ├── normal/
    ├── overhang/
    ├── underhang/
    └── vertical-misalignment/
```

Then run:

```shell
bash utils/preprocess/mafaulda/mafaulda_preprocess.sh
```

If you run the shell script above, you do not need to run the Python commands below.

To run `mafaulda_csv2wav_5s.py` directly and generate both datasets, run:

```shell
python utils/preprocess/mafaulda/mafaulda_csv2wav_5s.py --input_dir datasets_raw/mafaulda --output_dir datasets_wav/mafaulda_sound --dataset sound
python utils/preprocess/mafaulda/mafaulda_csv2wav_5s.py --input_dir datasets_raw/mafaulda --output_dir datasets_wav/mafaulda_vib --dataset vibration
```
