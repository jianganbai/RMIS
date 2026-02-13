This README describes the preprocessing scripts under `utils/preprocess` for converting raw datasets into no more than 10s wav files.

The RMIS benchmark unifies the forms of different signals as no more than 10s wav files. You can use `utils/preprocess/preprocess_raw_data.sh` to preprocess the raw data. This script is for IICA, WTPG, MaFaulDa, SDUST, UMGED, and PU; IIEE and DCASE are already 10s wav files and only need extraction. If you only want to preprocess a single dataset, refer to the dataset-specific README under `utils/preprocess/<dataset>/README.md`.

Preprocess via:

```shell
bash utils/preprocess/preprocess_raw_data.sh [iica mafaulda pu sdust_bearing sdust_gear umged wtpg]
```

Before running `utils/preprocess/preprocess_raw_data.sh`, the expected structure is:

```
datasets_raw/
├── iica/
│   ├── *.zip
│   ├── *.zip
│   └── *.zip
├── mafaulda/
│   └── *.zip
├── pu/
│   ├── K001.rar
│   ├── K002.rar
│   ├── K003.rar
│   ├── K004.rar
│   └── ...
├── sdust/
│   ├── 轴承数据集/
│   └── 齿轮数据集/
├── umged/
│   ├── G1/
│   └── G2/
└── wtpg/
    ├── broken/
    ├── healthy/
    ├── missing_tooth/
    ├── root_crack/
    └── wear/
```

After running `utils/preprocess/preprocess_raw_data.sh`, the expected structure is:

```
datasets_wav/
├── iica/
├── mafaulda_sound/
├── mafaulda_vib/
├── pu_cur/
├── pu_vib/
├── sdust_bearing/
├── sdust_gear/
├── umged_cur/
├── umged_sound/
├── umged_vib/
├── umged_vol/
└── wtpg/
```
