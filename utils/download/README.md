# Manual Downloading RMIS data

This README describes the raw data download options under `utils/download`.

For datasets hosted on Zenodo (DCASE 2020-2025, IICA, IIEE), you can use a browser, `zenodo_get`, or aria2-based downloaders (User-Agent, Referer, and Cookie are required for downloaders).

For datasets hosted on GitHub (SDUST), download via:

```shell
git clone https://github.com/JRWang-SDUST/SDUST-Dataset
```

For MaFaulDa and PU, you can use `utils/download/mafaulda_pu_downloader.py` with resumable downloads, state management, and concurrent workers.

Run:

```shell
python utils/download/mafaulda_pu_downloader.py pu mafaulda
```

The default output structure looks like:

```
datasets_raw/
├── download_state.json
├── mafaulda/
│   └── full.zip
└── pu/
    ├── K001.rar
    ├── K002.rar
    └── ...
```

You can refer to `utils/download/download_urls.yaml` or the URL tables below.

| DCASE Datasets | Dev Data | Eval Train | Eval Test |
| :--- | :--- | :--- | :--- |
| **DCASE 2020** | https://zenodo.org/records/3678171 | https://zenodo.org/records/3727685 | https://zenodo.org/records/3841772 |
| **DCASE 2021** | https://zenodo.org/records/4562016 | https://zenodo.org/records/4660992 | https://zenodo.org/records/4884786 |
| **DCASE 2022** | https://zenodo.org/records/6355122 | https://zenodo.org/records/6462969 | https://zenodo.org/records/6586456 |
| **DCASE 2023** | https://zenodo.org/records/7882613 | https://zenodo.org/records/7830345 | https://zenodo.org/records/7860847 |
| **DCASE 2024** | https://zenodo.org/records/10902294 | https://zenodo.org/records/11259435 | https://zenodo.org/records/11363076 |
| **DCASE 2025** | https://zenodo.org/records/15097779 | https://zenodo.org/records/15392814 | https://zenodo.org/records/15519362 |

| Other Datasets | URL |
| :--- | :--- |
| **IICA** | https://zenodo.org/records/7551606 |
| **IIEE** | https://zenodo.org/records/7551261 |
| **WTPG** | https://github.com/Liudd-BJUT/WT-planetary-gearbox-dataset |
| **MaFaulDa** | https://www02.smt.ufrj.br/~offshore/mfs/database/mafaulda/full.zip |
| **SDUST** | https://github.com/JRWang-SDUST/SDUST-Dataset |
| **UMGED** | https://github.com/LeeJMJM/UM-GearEccDataset |
| **PU** | https://groups.uni-paderborn.de/kat/BearingDataCenter/ |
