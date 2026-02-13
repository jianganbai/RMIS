WTPG preprocessing does not require extraction because the dataset provider ships raw files without compression.

Before running `wtpg_preprocess.sh`, the expected structure is:

```
datasets_raw/
└── wtpg/
    ├── broken/
    ├── healthy/
    ├── missing_tooth/
    ├── root_crack/
    └── wear/
```

WTPG preprocessing converts MAT files into FLAC files and then splits them into 10s segments.

Then run:

```shell
bash utils/preprocess/wtpg/wtpg_preprocess.sh
```

If you run the shell script above, you do not need to run the Python commands below.

To convert MAT files to FLAC files directly, run:

```shell
python utils/preprocess/wtpg/wtpg_MAT2flac.py --input_dir datasets_raw/wtpg --output_dir datasets_temp/wtpg
```

To split WTPG FLAC files into 10s segments directly, run:

```shell
python utils/preprocess/wtpg/wtpg_split_10.py --in_dir datasets_temp/wtpg --out_dir datasets_wav/wtpg
```
