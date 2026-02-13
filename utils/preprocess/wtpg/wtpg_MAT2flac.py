import os
import glob
import numpy as np
import soundfile as sf
from scipy.io import loadmat
import argparse
import multiprocessing as mp
from tqdm import tqdm
import math
import sys  # Import the sys module to use sys.exit() for program termination


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

        df = loadmat(mat_file)

        # Extract signal data
        try:
            # Handle possible nested array structure
            df = df['Data']
            df = df.T
        except KeyError as e:
            print(f"KeyError while processing {mat_file}: {e}")
            sys.exit(1)  # Terminate the program due to key error
        except IndexError as e:
            print(f"IndexError while processing {mat_file}: {e}")
            sys.exit(1)  # Terminate the program due to index error

        # Check the signal shape (number of channels)
        if df.shape[0] != 4:
            print(f"Unexpected number of channels (rows) in {mat_file}. Expected 4, got {df.shape[0]}")
            sys.exit(1)  # Terminate the program due to unexpected shape

        # Iterate through all channels to calculate the global maximum value of the data
        for channel in channels:
            try:
                data = df[int(channel) - 1, :]  # Get data for this channel
                data = data.astype('float64')  # Convert to float64
                local_max = max(local_max, np.max(np.abs(data)))  # Find the maximum absolute value
            except KeyError:
                print(f"Channel {channel} not found in file {mat_file}")
                sys.exit(1)  # Terminate the program if channel is not found

        # Update progress after processing each file
        progress_queue.put(1)

    return local_max


def mat_to_flac(mat_file, input_path, output_path, channels, global_max):
    """
    Reads a MAT file and converts data from specified channels into FLAC files.
    Normalizes the data using the global maximum value.
    """

    df = loadmat(mat_file)

    # Extract signal data
    try:
        # Handle possible nested array structure
        df = df['Data']
        df = df.T
    except KeyError as e:
        print(f"KeyError while processing {mat_file}: {e}")
        sys.exit(1)  # Terminate the program due to key error
    except IndexError as e:
        print(f"IndexError while processing {mat_file}: {e}")
        sys.exit(1)  # Terminate the program due to index error

    # Check the signal shape (number of channels)
    if df.shape[0] != 4:
        print(f"Unexpected number of channels (rows) in {mat_file}. Expected 4, got {df.shape[0]}")
        sys.exit(1)  # Terminate the program due to unexpected shape

    # Extract data from specified channels
    for channel in channels:
        try:
            data = df[int(channel) - 1, :]  # Get data for this channel
            data = data.astype('float64')  # Convert to float64

            # Normalize using the global maximum value
            data = normalize_signal(data, global_max)

            # Construct the output file path
            relative_path = os.path.relpath(mat_file, input_path)  # Get the relative path
            relative_path = os.path.splitext(relative_path)[0]  # Remove the .mat extension
            relative_path = relative_path.replace('/', '_').replace('\\', '_').replace(' ', '_')  # Replace path separators

            flac_filename = f"{relative_path}_channel{channel}.flac"
            flac_path = os.path.join(output_path, flac_filename)

            # Ensure the output directory exists
            os.makedirs(os.path.dirname(flac_path), exist_ok=True)

            # Write to a FLAC file using the soundfile library
            sf.write(flac_path, data, 48000)  # Sample rate 48 kHz
        except KeyError:
            print(f"Channel {channel} not found in file {mat_file}")
            sys.exit(1)  # Terminate the program if channel is not found


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
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default="datasets_raw/wtpg")
    parser.add_argument('--output_dir', type=str, default="datasets_temp/wtpg")
    parser.add_argument('--channel', type=str, default='1,2')
    parser.add_argument('--num_proc', type=int, default=20)
    args = parser.parse_args()

    # Recursively get all .MAT file paths using glob
    mat_files = glob.glob(os.path.join(args.input_dir, '**', '*.MAT'), recursive=True)

    # Get all channels
    channels = args.channel.split(',')

    # Calculate the global maximum value across all channels using multiprocessing
    global_max = find_global_max(mat_files, channels, args.num_proc)
    # Print the global maximum value
    print(f'global_max: {global_max}')

    # Initialize the progress bar for the conversion step
    pbar = tqdm(total=len(mat_files), desc="Converting MAT to FLAC")
    # Create a process pool
    p = mp.Pool(args.num_proc)
    for file in mat_files:
        p.apply_async(
            func=mat_to_flac,
            args=(file, args.input_dir, args.output_dir, channels, global_max),
            callback=lambda *args: pbar.update()  # Update progress bar upon task completion
        )
    p.close()
    p.join()
