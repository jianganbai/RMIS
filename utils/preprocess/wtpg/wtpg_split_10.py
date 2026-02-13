import os
import soundfile
import numpy as np
import multiprocessing as mp
import argparse

from tqdm import tqdm


def split_audio(file: str, out_dir: str) -> None:
    x, sr = soundfile.read(file)  # (num_frame) or (num_frame, num_channel)
    fn = os.path.splitext((os.path.basename(file)))[0]

    if x.ndim == 1:
        x = x[:, np.newaxis]
    for ch in range(x.shape[-1]):
        if tar_len is None:
            soundfile.write(
                os.path.join(out_dir, f'{fn}_ch{ch}.{ext}'),
                x[:, ch],
                samplerate=sr
            )
        else:
            seg_len = int(sr * tar_len)
            for i, start in enumerate(range(0, x.shape[0], seg_len)):
                end = start + seg_len
                if x.shape[0] >= end:
                    seg = x[start: end, ch]
                elif x.shape[0] - start > seg_len / 2:
                    seg = x[-seg_len:, ch]
                else:
                    break
                soundfile.write(
                    os.path.join(out_dir, f'{fn}_ch{ch}_{i}.{ext}'),
                    seg,
                    samplerate=sr
                )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', type=str, default="datasets_temp/wtpg")
    parser.add_argument('--out_dir', type=str, default="datasets_wav/wtpg")
    args = parser.parse_args()

    ext = 'flac'
    tar_len = 10  # second
    num_proc = 20

    file_list = []
    for dirpath, dirnames, filenames in os.walk(args.in_dir):
        for filename in filenames:
            if filename.endswith(ext):
                file_list.append(os.path.join(dirpath, filename))

    os.makedirs(args.out_dir, exist_ok=True)

    pbar = tqdm(total=len(file_list))
    p = mp.Pool(num_proc)
    for file in file_list:
        p.apply_async(
            func=split_audio,
            args=(file, args.out_dir),
            callback=lambda *args: pbar.update()
        )
    p.close()
    p.join()
