IICA preprocessing extracts the archives and splits audio into 10s wav segments stored in `datasets_wav/iica`.

Before running `iica_preprocess.sh`, the expected structure is:

```
datasets_raw/
└── iica/
    ├── *.zip
    ├── *.zip
    └── *.zip
```

After running `iica_preprocess.sh`, the expected structure is:

```
datasets_raw/
└── iica/
    ├── tubeleak/
    ├── ventleak/
    └── ventlow/
```

Then run:

```shell
bash utils/preprocess/iica/iica_preprocess.sh
```

If you run the shell script above, you do not need to run the Python command below.

To split IICA wav files into 10s segments directly, run:

```shell
python utils/preprocess/iica/iica_split_10.py --in_dir datasets_raw/iica --out_dir datasets_wav/iica
```
