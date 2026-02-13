PU preprocessing extracts the archives, organizes the folder structure, and generates wav datasets in `datasets_wav/pu_cur` and `datasets_wav/pu_vib`.

Before running `pu_preprocess.sh`, the expected structure is:

```
datasets_raw/
└── pu/
    ├── K001.rar
    ├── K002.rar
    ├── K003.rar
    ├── K004.rar
    └── ...
```

After running `pu_preprocess.sh`, the expected structure is:

```
datasets_raw/
└── pu/
    ├── healthy/
    ├── IR/
    └── OR/
```

PU preprocessing extracts and organizes the dataset, fixes the KA08 MAT file, and converts MAT files into wav files.

Install 7z or unrar before running the shell script. If you already extracted the rar files with WinRAR, you can still run the script; it will skip extraction and continue with the organization and conversion steps.

Windows (Bash):

Install 7-Zip from https://www.7-zip.org/ and add it to PATH so `7z` works in Bash.
Install UnRAR for Windows from https://www.rarlab.com/rar_add.htm and add it to PATH so `unrar` works in Bash.

Linux:

```shell
sudo apt-get update
sudo apt-get install -y p7zip-full unrar
```

macOS:

```shell
brew install p7zip unrar
```

Then run:

```shell
bash utils/preprocess/pu/pu_preprocess.sh
```

If you run the shell script above, you do not need to run the Python commands below.

To convert PU MAT files to wav files directly, run:

```shell
python utils/preprocess/pu/pu_mat2wav_4s.py --input_dir datasets_raw/pu --output_dir datasets_wav/pu_cur --dataset current
python utils/preprocess/pu/pu_mat2wav_4s.py --input_dir datasets_raw/pu --output_dir datasets_wav/pu_vib --dataset vibration
```
