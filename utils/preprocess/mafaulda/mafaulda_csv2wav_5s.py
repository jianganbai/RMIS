import os
import glob
import numpy as np
import soundfile as sf
import pandas as pd
import argparse
import multiprocessing as mp
from tqdm import tqdm
import math
import sys


def normalize_signal(signal, global_max):
    """
    Normalizes the signal.
    If the global maximum is greater than 1.0, normalize the signal,
    otherwise return the original signal.
    """
    if global_max > 1.0:
        return signal / global_max
    else:
        return signal


def find_local_max(csv_files, channels, progress_queue):
    """
    Each subprocess calculates the maximum value for its assigned set of files (across all channels).
    Returns the local maximum value and updates progress via a queue.
    """
    local_max = 0  # Records the local maximum value

    for csv_file in csv_files:

        df = pd.read_csv(csv_file, header=None)

        # Ensure the CSV file has at least 8 columns
        if df.shape[1] < 8:
            print(f"File {csv_file} has insufficient columns, skipping.")
            sys.exit(1)  # Terminate the program due to insufficient columns

        # Iterate through all channels to calculate the global maximum value of the data
        for channel in channels:
            try:
                data = df.iloc[:, int(channel)-1].values  # Get data from the specified column
                data = data.astype('float32')
                local_max = max(local_max, np.max(np.abs(data)))  # Find the maximum value for this channel's data
            except KeyError:
                print(f"Channel {channel} not found in file {csv_file}")
                sys.exit(1)  # Terminate the program if the channel is not found

        # Update progress after processing each file
        progress_queue.put(1)

    return local_max


def csv_to_wav(csv_file, input_path, output_path, channels, global_max):
    """
    Reads a CSV file and converts data from specified channels into WAV files.
    Normalizes the data using the global maximum value.
    """

    df = pd.read_csv(csv_file, header=None)

    # Ensure the CSV file has at least 8 columns
    if df.shape[1] < 8:
        print(f"File {csv_file} has insufficient columns, skipping.")
        sys.exit(1)  # Terminate the program due to insufficient columns

    # Extract data from the specified channels
    for channel in channels:
        try:
            data = df.iloc[:, int(channel)-1].values  # Get data from the specified column
            data = data.astype('float32')

            # Normalize using the global maximum value
            data = normalize_signal(data, global_max)

            # Construct the output file path
            relative_path = os.path.relpath(csv_file, input_path)  # Get the relative path
            relative_path = os.path.splitext(relative_path)[0]  # Remove the .csv extension
            relative_path = relative_path.replace('/', '_').replace('\\', '_').replace(' ', '_')  # Replace path separators

            wav_filename = f"{relative_path}_channel{channel}.wav"
            wav_path = os.path.join(output_path, wav_filename)

            # Ensure the output directory exists
            os.makedirs(os.path.dirname(wav_path), exist_ok=True)

            # Write to a WAV file using the soundfile library
            sf.write(wav_path, data, 50000)  # Sample rate 50 kHz
        except KeyError:
            print(f"Channel {channel} not found in file {csv_file}")
            sys.exit(1)  # Terminate the program if the channel is not found


def find_global_max(csv_files, channels, num_proc):
    """
    Calculates the global maximum value across all channels using multiprocessing,
    with progress bar updates.
    """
    # If the number of files is less than the number of processes, adjust the process count
    num_proc = min(num_proc, len(csv_files))

    # Split csv_files into chunks for each process
    chunk_size = math.ceil(len(csv_files) / num_proc)  # Ensure each process handles at least one file
    file_chunks = [csv_files[i:i + chunk_size] for i in range(0, len(csv_files), chunk_size)]

    # Create a process pool
    pool = mp.Pool(num_proc)

    # Create a progress queue
    progress_queue = mp.Manager().Queue()

    # Each process calculates its local maximum
    results = [pool.apply_async(find_local_max, args=(chunk, channels, progress_queue)) for chunk in file_chunks]

    # Initialize the progress bar
    pbar = tqdm(total=len(csv_files), desc="Calculating Global Max")

    # Listen to the progress queue and update the progress bar
    for _ in range(len(csv_files)):
        progress_queue.get()  # Get data from the queue (indicates a file is processed)
        pbar.update(1)  # Update the progress bar

    # Get results from all processes and merge them
    global_max = 0
    for result in results:
        local_max = result.get()  # Get the local maximum from each subprocess
        global_max = max(global_max, local_max)  # Determine the global maximum from all subprocesses

    pool.close()
    pool.join()

    pbar.close()  # Close the progress bar

    return global_max


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert MaFaulDa CSV files to WAV format based on channel selection.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--input_dir', type=str, default="datasets_raw/mafaulda")
    parser.add_argument('--output_dir', type=str, default="datasets_wav/mafaulda_vib")
    parser.add_argument('--num_proc', type=int, default=20)

    # Create a mutually exclusive group for channel selection modes.
    # This ensures the user can only use one mode at a time.
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--channel', type=str,
                            help="Mode 'channel': Specify channels directly, e.g., '8' or '2,3,4,5,6,7'.")
    mode_group.add_argument('--dataset', type=str, choices=['sound', 'vibration'],
                            help="Mode 'dataset': Select a predefined dataset: 'sound' (channel 8) or 'vibration' (channels 2-7).")

    args = parser.parse_args()

    # Determine which channels to process based on the selected mode.
    channels = []
    if args.channel:
        # User selected 'channel' mode.
        print(f"Mode selected: channel. Custom channels: {args.channel}")
        channels = args.channel.split(',')
    elif args.dataset:
        # User selected 'dataset' mode.
        print(f"Mode selected: dataset. Dataset name: {args.dataset}")
        if args.dataset == 'sound':
            channels = ['8']
        elif args.dataset == 'vibration':
            channels = [str(i) for i in range(2, 8)]  # Channels 2, 3, 4, 5, 6, 7
    else:
        # Default behavior if no mode is specified: select sound (channel 8).
        print("No mode specified. Defaulting to 'sound' dataset (channel 8).")
        channels = ['8']

    print(f"Final channels to be processed: {', '.join(channels)}")

    # Recursively get all .csv file paths using glob
    csv_files = glob.glob(os.path.join(args.input_dir, '**', '*.csv'), recursive=True)

    if not csv_files:
        print(f"Error: No .csv files found in {args.input_dir}. Please check the directory path.")
        sys.exit(1)

    # Calculate the global maximum value across all channels using multiprocessing
    global_max = find_global_max(csv_files, channels, args.num_proc)
    # Print the global maximum value
    print(f'global_max: {global_max}')

    # Initialize the progress bar for the conversion step
    pbar = tqdm(total=len(csv_files), desc="Converting CSV to WAV")
    # Create a process pool
    p = mp.Pool(args.num_proc)
    for file in csv_files:
        p.apply_async(
            func=csv_to_wav,
            args=(file, args.input_dir, args.output_dir, channels, global_max),
            callback=lambda *args: pbar.update()  # Update progress bar upon task completion
        )
    p.close()
    p.join()
