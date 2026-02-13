import asyncio
import httpx
import argparse
import sys
import json
from pathlib import Path
import aiofiles
from tqdm.asyncio import tqdm
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# --- 1. Dataset Metadata & Constants ---
DATASET_MAP = {
    "pu": "https://groups.uni-paderborn.de/kat/BearingDataCenter/",
    "mafaulda": "https://www02.smt.ufrj.br/~offshore/mfs/database/mafaulda/full.zip",
}
CHUNK_SIZE = 1024 * 1024
ARCHIVE_EXTENSIONS = ('.zip', '.rar', '.tar', '.gz', '.bz2', '.7z')


# --- 2. State Management Functions ---
def load_state(state_file_path: Path) -> list:
    """Load the download plan from the state file if it exists."""
    if state_file_path.exists():
        with open(state_file_path, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                print("Warning: State file is corrupted. Starting fresh.", file=sys.stderr)
                return []
    return []


def save_state(state: list, state_file_path: Path):
    """Save the current state of the download plan."""
    with open(state_file_path, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=4)


def initialize_state(plan: list[tuple[str, str]]) -> list[dict]:
    """Convert the initial plan to the structured, platform-agnostic state format."""
    state = []
    for name, url in plan:
        state.append({
            "dataset_name": name,
            "url": url,
            "status": "not_started",
            "total_size": 0,
        })
    return state


# --- 3. Helper & Discovery Functions ---
def is_direct_download_link(url: str) -> bool:
    return url.lower().endswith(ARCHIVE_EXTENSIONS)


def is_archive_file(url: str) -> bool:
    return url.lower().endswith(ARCHIVE_EXTENSIONS)


def is_sub_directory(url: str, base_url: str) -> bool:
    return url.startswith(base_url) and not is_archive_file(url)


async def discover_files_recursive(client: httpx.AsyncClient, base_url: str, visited_urls: set = None) -> list[str]:
    if visited_urls is None:
        visited_urls = set()
    normalized_url = base_url.rstrip('/')
    if normalized_url in visited_urls:
        return []
    visited_urls.add(normalized_url)
    found_files = []
    print(f"Scanning directory: {base_url}")
    try:
        response = await client.get(base_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'lxml')
        for link in soup.find_all('a', href=True):
            full_url = urljoin(base_url, link['href'])
            if is_archive_file(full_url) and full_url not in found_files:
                print(f"  - Found file: {full_url.split('/')[-1]}")
                found_files.append(full_url)
            elif is_sub_directory(full_url, DATASET_MAP['pu']) and full_url not in visited_urls:
                found_files.extend(await discover_files_recursive(client, full_url, visited_urls))
    except httpx.RequestError as e:
        print(f"Error scanning URL {base_url}: {e}", file=sys.stderr)
    return found_files


async def generate_download_plan(datasets_to_process: list[str]) -> list[tuple[str, str]]:
    plan = []
    async with httpx.AsyncClient() as client:
        for name in datasets_to_process:
            url_or_list = DATASET_MAP.get(name)
            if not url_or_list:
                print(f"Warning: Dataset '{name}' not found. Skipping.", file=sys.stderr)
                continue
            if isinstance(url_or_list, list):
                print(f"'{name}': Found a manual list of {len(url_or_list)} file links.")
                for file_url in url_or_list:
                    plan.append((name, file_url))
            elif is_direct_download_link(url_or_list):
                print(f"'{name}': Found direct file link.")
                plan.append((name, url_or_list))
            else:
                print(f"'{name}': URL is a directory, starting scan...")
                discovered_files = await discover_files_recursive(client, url_or_list)
                for file_url in discovered_files:
                    plan.append((name, file_url))
    return plan


# --- 4. Core Download Function ---
async def download_file_with_resume(client: httpx.AsyncClient, url: str, destination_path: Path, item: dict, full_plan: list, state_file_path: Path):
    downloaded_size = 0
    if destination_path.exists():
        downloaded_size = destination_path.stat().st_size
    headers = {'Range': f'bytes={downloaded_size}-'} if downloaded_size > 0 else {}
    try:
        async with client.stream("GET", url, headers=headers) as response:
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            is_resuming = response.status_code == 206
            total_expected_size = downloaded_size + total_size if is_resuming else total_size
            if not is_resuming:
                downloaded_size = 0

            if item['total_size'] != total_expected_size:
                item['total_size'] = total_expected_size
                save_state(full_plan, state_file_path)

            pbar = tqdm(total=total_expected_size, initial=downloaded_size, unit='B', unit_scale=True, desc=destination_path.name, ncols=80)
            file_mode = 'ab' if is_resuming and downloaded_size > 0 else 'wb'
            async with aiofiles.open(destination_path, file_mode) as f:
                async for chunk in response.aiter_bytes(chunk_size=CHUNK_SIZE):
                    await f.write(chunk)
                    pbar.update(len(chunk))
            pbar.close()
            final_size = destination_path.stat().st_size
            if total_expected_size != 0 and final_size != total_expected_size:
                print(f"Error: File size mismatch for {destination_path.name}.", file=sys.stderr)
            else:
                print(f"{destination_path.name} downloaded successfully.")
    except (httpx.RequestError, IOError) as e:
        print(f"Error during download for {destination_path.name}: {e}", file=sys.stderr)
        raise
    except asyncio.CancelledError:
        print(f"Download cancelled for {destination_path.name}.")
        raise


# --- 5. Task Worker ---
async def download_worker(client: httpx.AsyncClient, semaphore: asyncio.Semaphore, item: dict, full_plan: list, state_file_path: Path, output_dir: Path):
    async with semaphore:
        try:
            dataset_name = item['dataset_name']
            filename = item['url'].split('/')[-1]
            destination_path = output_dir / dataset_name / filename
            destination_path.parent.mkdir(parents=True, exist_ok=True)

            item['status'] = 'downloading'
            save_state(full_plan, state_file_path)

            if destination_path.exists():
                print(f"File {destination_path.name} already exists. Verifying.")

            await download_file_with_resume(client, item['url'], destination_path, item, full_plan, state_file_path)

            item['status'] = 'completed'
            save_state(full_plan, state_file_path)
        except Exception:
            item['status'] = 'failed'
            save_state(full_plan, state_file_path)
            print(f"Download failed for {item['url']}. Marked as failed.", file=sys.stderr)


# --- 6. Function to Merge Plans ---
async def update_state_with_new_datasets(datasets_to_process: list[str], current_plan: list[dict]) -> list[dict]:
    existing_datasets = {item['dataset_name'] for item in current_plan}
    new_datasets_to_scan = [name for name in datasets_to_process if name not in existing_datasets]

    if not new_datasets_to_scan:
        return current_plan

    print(f"New datasets detected: {new_datasets_to_scan}. Scanning for files...")
    newly_discovered_plan = await generate_download_plan(new_datasets_to_scan)
    new_state_items = initialize_state(newly_discovered_plan)
    current_plan.extend(new_state_items)
    return current_plan


# --- 7. Main Function ---
async def main():
    parser = argparse.ArgumentParser(description="A robust, resumable asyncio script to download datasets.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('datasets', metavar='NAME', type=str, nargs='+', choices=DATASET_MAP.keys(), help="Names of datasets.")
    parser.add_argument('--output_dir', type=str, default='./datasets_raw', help="Directory to store datasets")
    parser.add_argument('--workers', type=int, default=10, help="Number of concurrent download tasks")
    parser.add_argument('--force-rescan', action='store_true', help="Force re-scanning all specified URLs.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_file_path = output_dir / "download_state.json"

    print(f"Output directory: {output_dir.resolve()}")
    print(f"State file: {state_file_path.resolve()}")

    download_plan = []
    if not args.force_rescan:
        download_plan = load_state(state_file_path)

    if not download_plan or args.force_rescan:
        print("--- Phase 1: No state file or rescan forced. Discovering files... ---")
        if args.force_rescan:
            download_plan = []
        initial_plan = await generate_download_plan(args.datasets)
        if not initial_plan:
            print("No files found to download. Exiting.")
            return
        download_plan = initialize_state(initial_plan)
        save_state(download_plan, state_file_path)
        print("Discovery complete. State file created/overwritten.")
    else:
        print("--- Found existing state file. Checking for new datasets to add... ---")
        download_plan = await update_state_with_new_datasets(args.datasets, download_plan)
        save_state(download_plan, state_file_path)

    tasks_to_run = [
        item for item in download_plan
        if item['dataset_name'] in args.datasets and item['status'] != 'completed'
    ]
    if not tasks_to_run:
        print("All files are already marked as completed.")
        return

    print(f"--- Phase 2: Starting download for {len(tasks_to_run)} of {len(download_plan)} total files ---")
    print("Press Ctrl+C to stop gracefully.")
    print("-" * 50)

    semaphore = asyncio.Semaphore(args.workers)
    async with httpx.AsyncClient(timeout=None) as client:
        tasks = [download_worker(client, semaphore, item, download_plan, state_file_path, output_dir) for item in tasks_to_run]
        await asyncio.gather(*tasks)

    print("-" * 50)
    completed_count = sum(1 for item in download_plan if item['status'] == 'completed')
    print(f"Process finished. {completed_count}/{len(download_plan)} files are completed.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Script interrupted by user. State has been saved. You can resume later.")
