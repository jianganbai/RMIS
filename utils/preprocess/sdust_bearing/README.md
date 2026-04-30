SDUST bearing preprocessing does not require extraction because the dataset provider ships raw files without compression.

Before running `sdust_bearing_preprocess.sh`, the expected structure is:

```
datasets_raw/
└── sdust/
    └── 轴承数据集/
```

After running `sdust_bearing_preprocess.sh`, the expected structure is:

```
datasets_raw/
└── sdust/
    └── bearing/
```

SDUST bearing preprocessing converts MAT files into wav files and then splits them into 10s segments.

Then run:

```shell
bash utils/preprocess/sdust_bearing/sdust_bearing_preprocess.sh
```

If you run the shell script above, you do not need to run the Python commands below.

To convert MAT files to wav files directly, run:

```shell
python utils/preprocess/sdust_bearing/sdust_bearing_mat2wav_40s.py --input_dir datasets_raw/sdust/bearing --output_dir datasets_temp/sdust_bearing
```

To split SDUST bearing wav files into 10s segments directly, run:

```shell
python utils/preprocess/sdust_bearing/sdust_bearing_split_10.py --in_dir datasets_temp/sdust_bearing --out_dir datasets_wav/sdust_bearing
```
