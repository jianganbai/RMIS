import os
import soundfile
import numpy as np
import multiprocessing as mp
import argparse
from tqdm import tqdm


def split_audio(file: str, specific_out_dir: str, tar_len: float, ext: str) -> None:
    """
    Splits a single audio file into segments of a target length.

    Args:
        file (str): Path to the input audio file.
        specific_out_dir (str): The specific output directory for this file's segments,
                                which mirrors the original directory structure.
        tar_len (float): Target length of each segment in seconds.
        ext (str): The file extension for the output segments (e.g., 'wav').
    """
    try:
        # Before writing, ensure the specific output directory for this file exists.
        # This is safe to call even if the directory already exists.
        os.makedirs(specific_out_dir, exist_ok=True)

        x, sr = soundfile.read(file)  # (num_frame) or (num_frame, num_channel)
        fn = os.path.splitext(os.path.basename(file))[0]

        if x.ndim == 1:
            x = x[:, np.newaxis]

        for ch in range(x.shape[-1]):
            # If tar_len is not specified, just split channels without segmenting.
            if tar_len is None:
                soundfile.write(
                    os.path.join(specific_out_dir, f'{fn}_ch{ch}.{ext}'),
                    x[:, ch],
                    samplerate=sr
                )
            # Otherwise, segment the audio into chunks of tar_len seconds.
            else:
                seg_len = int(sr * tar_len)
                for i, start in enumerate(range(0, x.shape[0], seg_len)):
                    end = start + seg_len
                    if x.shape[0] >= end:
                        seg = x[start:end, ch]
                    # Handle the last segment: if it's more than half the target length, keep it.
                    elif x.shape[0] - start > seg_len / 2:
                        seg = x[-seg_len:, ch]
                    else:
                        break  # The last segment is too short, discard it.
                    soundfile.write(
                        os.path.join(specific_out_dir, f'{fn}{i}.{ext}'),
                        seg,
                        samplerate=sr
                    )
    except Exception as e:
        print(f"Error processing file {file}: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Split multi-channel audio files into segments while preserving directory structure."
    )
    parser.add_argument('--in_dir', type=str, default="datasets_raw/iica",
                        help="Root directory of the input dataset.")
    parser.add_argument('--out_dir', type=str, default="datasets_wav/iica",
                        help="Root directory for the output split dataset.")
    args = parser.parse_args()

    ext = 'wav'
    tar_len = 10  # second
    num_proc = 20

    tasks = []

    for dirpath, _, filenames in os.walk(args.in_dir):
        for filename in filenames:
            # Process only files with the specified extension, case-insensitively.
            if filename.lower().endswith(ext):
                full_input_path = os.path.join(dirpath, filename)

                relative_dir = os.path.relpath(dirpath, args.in_dir)

                specific_out_dir = os.path.join(args.out_dir, relative_dir)
                tasks.append((full_input_path, specific_out_dir))
    if not tasks:
        print(f"No '.{ext}' files found in '{args.in_dir}'. Exiting.")
        exit()

    print(f"Found {len(tasks)} files to process. Starting parallel processing...")

    os.makedirs(args.out_dir, exist_ok=True)

    pbar = tqdm(total=len(tasks))
    p = mp.Pool(num_proc)

    # Pass the specific output directory to each worker.
    for input_file, specific_output_dir in tasks:
        p.apply_async(
            func=split_audio,
            args=(input_file, specific_output_dir, tar_len, ext),
            callback=lambda *args: pbar.update()
        )
    p.close()
    p.join()
    pbar.close()

    print("All processing completed.")
