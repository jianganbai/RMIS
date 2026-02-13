import os
import glob
import numpy as np
import soundfile as sf
import hdf5storage
import argparse
import multiprocessing as mp
from tqdm import tqdm
import math
import sys  # Import sys module to use sys.exit() for program termination


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


def find_local_max(mat_files, channels, progress_queue):
    """
    Each subprocess calculates the maximum value for its assigned set of files (across all channels).
    Returns the local maximum value and updates progress via a queue.
    """
    local_max = 0  # Records the local maximum value

    for mat_file in mat_files:

        df = hdf5storage.loadmat(mat_file)

        # Iterate through all channels to calculate the global maximum value of the data
        for channel in channels:
            try:
                # Assuming the key format is 'DataSensorXX'
                data = df[f'DataSensor{channel}']
                data = data.astype('float32')  # Convert to float32
                local_max = max(local_max, np.max(np.abs(data)))  # Find the maximum value for this channel's data
            except KeyError:
                print(f"Channel {channel} (Key: DataSensor{channel}) not found in file {mat_file}")
                sys.exit(1)  # Terminate the program if the channel is not found

        # Update progress after processing each file
        progress_queue.put(1)

    return local_max


def mat_to_wav(mat_file, input_path, output_path, channels, global_max):
    """
    Reads a MAT file and converts data from specified channels into WAV files.
    Normalizes the data using the global maximum value.
    """

    df = hdf5storage.loadmat(mat_file)

    # Ensure the MAT file has enough keys/data (original logic checked length < 11)
    if len(df) < 11:
        print(f"File {mat_file} has insufficient keys/data, skipping.")
        return  # Skip if insufficient data

    # Extract data from the specified channels
    for channel in channels:
        try:
            data = df[f'DataSensor{channel}']  # Get data for the specific channel
            data = data.astype('float32')  # Convert to float32

            # Normalize using the global maximum value
            data = normalize_signal(data, global_max)

            # Construct the output file path
            relative_path = os.path.relpath(mat_file, input_path)  # Get the relative path
            relative_path = os.path.splitext(relative_path)[0]  # Remove the .mat extension
            relative_path = relative_path.replace('/', '_').replace('\\', '_').replace(' ', '_')  # Replace path separators

            wav_filename = f"{relative_path}_channel{channel}.wav"
            wav_path = os.path.join(output_path, wav_filename)

            # Ensure the output directory exists
            os.makedirs(os.path.dirname(wav_path), exist_ok=True)

            # Write to a WAV file using the soundfile library
            sf.write(wav_path, data, 51200)  # Sample rate 51.2 kHz
        except KeyError:
            print(f"Channel {channel} (Key: DataSensor{channel}) not found in file {mat_file}")
            sys.exit(1)  # Terminate the program if the channel is not found


def find_global_max(mat_files, channels, num_proc):
    """
    Calculates the global maximum value across all channels using multiprocessing,
    with progress bar updates.
    """
    # If the number of files is less than the number of processes, adjust the process count
    num_proc = min(num_proc, len(mat_files))

    # Split mat_files into chunks for each process
    chunk_size = math.ceil(len(mat_files) / num_proc)  # Ensure each process handles at least one file
    file_chunks = [mat_files[i:i + chunk_size] for i in range(0, len(mat_files), chunk_size)]

    # Create a process pool
    pool = mp.Pool(num_proc)

    # Create a progress queue
    progress_queue = mp.Manager().Queue()

    # Each process calculates its local maximum
    results = [pool.apply_async(find_local_max, args=(chunk, channels, progress_queue)) for chunk in file_chunks]

    # Initialize the progress bar
    pbar = tqdm(total=len(mat_files), desc="Calculating Global Max")

    # Listen to the progress queue and update the progress bar
    for _ in range(len(mat_files)):
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
        description="Convert UMGED dataset MAT files to WAV format.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--input_dir', type=str, default="datasets_raw/umged")
    parser.add_argument('--output_dir', type=str, default="datasets_temp/umged_vol")
    parser.add_argument('--num_proc', type=int, default=20)

    # Create a mutually exclusive group for channel selection modes.
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--channel', type=str,
                            help="Mode 'channel': Specify channels directly, e.g., '03,04,05'.")
    mode_group.add_argument('--dataset', type=str, choices=['current', 'voltage', 'vibration', 'sound'],
                            help="Mode 'dataset': Select a predefined dataset:\n"
                                 "  'current'   (channels 01,02)\n"
                                 "  'voltage'   (channels 07,08)\n"
                                 "  'vibration' (channels 03,04,05)\n"
                                 "  'sound'     (channel 06)")

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
        if args.dataset == 'current':
            channels = ['01', '02']
        elif args.dataset == 'voltage':
            channels = ['07', '08']
        elif args.dataset == 'vibration':
            channels = ['03', '04', '05']
        elif args.dataset == 'sound':
            channels = ['06']
    else:
        # Default behavior if no mode is specified: select vibration (03,04,05).
        print("No mode specified. Defaulting to 'vibration' dataset (channels 03,04,05).")
        channels = ['03', '04', '05']

    print(f"Final channels to be processed: {', '.join(channels)}")

    # Recursively get all .mat file paths using glob
    mat_files = glob.glob(os.path.join(args.input_dir, '**', '*.mat'), recursive=True)

    if not mat_files:
        print(f"Error: No .mat files found in {args.input_dir}. Please check the directory path.")
        sys.exit(1)

    # Calculate the global maximum value across all channels using multiprocessing
    global_max = find_global_max(mat_files, channels, args.num_proc)
    # Print the global maximum value
    print(f'global_max: {global_max}')

    # Initialize the progress bar for the conversion step
    pbar = tqdm(total=len(mat_files), desc="Converting MAT to WAV")
    # Create a process pool
    p = mp.Pool(args.num_proc)
    for file in mat_files:
        p.apply_async(
            func=mat_to_wav,
            args=(file, args.input_dir, args.output_dir, channels, global_max),
            callback=lambda *args: pbar.update()  # Update progress bar upon task completion
        )
    p.close()
    p.join()
