SDUST gear preprocessing does not require extraction because the dataset provider ships raw files without compression.

Before running `sdust_gear_preprocess.sh`, the expected structure is:

```
datasets_raw/
└── sdust/
    └── 齿轮数据集/
        ├── NC/
        ├── 太阳断裂/
        ├── 太阳点蚀/
        ├── 太阳磨损/
        ├── 行星断裂/
        ├── 行星点蚀/
        └── 行星磨损/
```

After running `sdust_gear_preprocess.sh`, the expected structure is:

```
datasets_raw/
└── sdust/
    └── gear/
        ├── NC/
        ├── sunfracture/
        ├── sunpitting/
        ├── sunwear/
        ├── planetrayfracture/
        ├── planetraypitting/
        └── planetraywear/
```

SDUST gear preprocessing converts MAT files into wav files and then splits them into 10s segments.

Then run:

```shell
bash utils/preprocess/sdust_gear/sdust_gear_preprocess.sh
```

If you run the shell script above, you do not need to run the Python commands below.

To convert MAT files to wav files directly, run:

```shell
python utils/preprocess/sdust_gear/sdust_gear_mat2wav_20s.py --input_dir datasets_raw/sdust/gear --output_dir datasets_temp/sdust_gear
```

To split SDUST gear wav files into 10s segments directly, run:

```shell
python utils/preprocess/sdust_gear/sdust_gear_split_10.py --in_dir datasets_temp/sdust_gear --out_dir datasets_wav/sdust_gear
```
