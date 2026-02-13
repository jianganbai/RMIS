UMGED preprocessing does not require extraction because the dataset provider ships raw files without compression.

Before running `umged_preprocess.sh`, the expected structure is:

```
datasets_raw/
└── umged/
    ├── G1/
    └── G2/
```

UMGED preprocessing converts MAT files into wav files and then splits them into 10s segments.

Then run:

```shell
bash utils/preprocess/umged/umged_preprocess.sh
```

If you run the shell script above, you do not need to run the Python commands below.

To convert MAT files to wav files directly, run:

```shell
python utils/preprocess/umged/umged_mat2wav_600s.py --input_dir datasets_raw/umged --output_dir datasets_temp/umged_sound --dataset sound
python utils/preprocess/umged/umged_mat2wav_600s.py --input_dir datasets_raw/umged --output_dir datasets_temp/umged_vib --dataset vibration
python utils/preprocess/umged/umged_mat2wav_600s.py --input_dir datasets_raw/umged --output_dir datasets_temp/umged_vol --dataset voltage
python utils/preprocess/umged/umged_mat2wav_600s.py --input_dir datasets_raw/umged --output_dir datasets_temp/umged_cur --dataset current
```

To split UMGED wav files into 10s segments directly, run:

```shell
python utils/preprocess/umged/umged_split_10.py --in_dir datasets_temp/umged_sound --out_dir datasets_wav/umged_sound
python utils/preprocess/umged/umged_split_10.py --in_dir datasets_temp/umged_vib --out_dir datasets_wav/umged_vib
python utils/preprocess/umged/umged_split_10.py --in_dir datasets_temp/umged_vol --out_dir datasets_wav/umged_vol
python utils/preprocess/umged/umged_split_10.py --in_dir datasets_temp/umged_cur --out_dir datasets_wav/umged_cur
```
