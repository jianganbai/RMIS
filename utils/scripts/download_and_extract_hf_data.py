import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from huggingface_hub import snapshot_download

DEFAULT_REPO_ID = "jiangab/RMIS"
DEFAULT_SUBSETS: List[str] = [
    "dcase20",
    "dcase21",
    "dcase22",
    "dcase23",
    "dcase24",
    "dcase25",
    "iica",
    "iiee",
    "mafaulda_sound",
    "mafaulda_vib",
    "pu_cur",
    "pu_vib",
    "sdust_bearing",
    "sdust_gear",
    "umged_cur",
    "umged_sound",
    "umged_vib",
    "umged_vol",
    "wtpg",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download RMIS parquet shards from Hugging Face and extract the "
            "embedded audio files as wav files under RMIS-style subset folders."
        )
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory that will contain subset folders such as dcase20/, iiee/, wtpg/, etc.",
    )
    parser.add_argument(
        "--repo_id",
        default=DEFAULT_REPO_ID,
        help=f"Hugging Face dataset repo id. Default: {DEFAULT_REPO_ID}",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional branch, tag, or commit hash.",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        help="Optional Hugging Face cache directory.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional Hugging Face token. Falls back to HF_TOKEN if omitted.",
    )
    parser.add_argument(
        "--subset",
        nargs="+",
        default=None,
        help="Subset list to process. Default is all RMIS subsets.",
    )
    parser.add_argument(
        "--force_redownload",
        action="store_true",
        help="Overwrite existing local parquet files even if file size matches.",
    )
    parser.add_argument(
        "--force_reextract",
        action="store_true",
        help="Ignore extraction markers and rewrite wav files.",
    )
    parser.add_argument(
        "--remove_parquet_after_extract",
        action="store_true",
        help="Delete local parquet shards after successful extraction.",
    )
    return parser.parse_args()


def validate_subsets(subsets: Sequence[str]) -> List[str]:
    unknown = sorted(set(subsets) - set(DEFAULT_SUBSETS))
    if unknown:
        raise ValueError(
            "Unknown subset(s): "
            + ", ".join(unknown)
            + ". Allowed values: "
            + ", ".join(DEFAULT_SUBSETS)
        )
    return list(subsets)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def clean_stale_temp_files(subset_dir: Path) -> None:
    for pattern in (".*.copying", ".*.writing", ".*.marker_tmp"):
        for path in subset_dir.glob(pattern):
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def copy_file_atomic(src: Path, dst: Path) -> None:
    tmp = dst.parent / f".{dst.name}.copying"
    if tmp.exists():
        tmp.unlink()
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def sync_subset_parquets(
    snapshot_root: Path,
    output_dir: Path,
    subset: str,
    force_redownload: bool,
) -> List[Path]:
    src_subset_dir = snapshot_root / subset
    if not src_subset_dir.exists():
        raise FileNotFoundError(f"Subset directory not found in snapshot: {src_subset_dir}")

    dst_subset_dir = output_dir / subset
    ensure_dir(dst_subset_dir)
    clean_stale_temp_files(dst_subset_dir)

    src_parquets = sorted(src_subset_dir.glob("*.parquet"))
    if not src_parquets:
        raise FileNotFoundError(f"No parquet files found for subset {subset} under {src_subset_dir}")

    dst_parquets: List[Path] = []
    for src in src_parquets:
        dst = dst_subset_dir / src.name
        same_size = dst.exists() and dst.stat().st_size == src.stat().st_size
        if force_redownload or not same_size:
            copy_file_atomic(src, dst)
            print(f"[SYNC] {src.name} -> {dst}")
        else:
            print(f"[SKIP] {dst.name} already present")
        dst_parquets.append(dst)

    return dst_parquets


def load_marker(marker_path: Path) -> Optional[Dict]:
    if not marker_path.exists():
        return None
    try:
        return json.loads(marker_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def marker_matches_source(marker: Dict, parquet_path: Path) -> bool:
    if marker.get("completed") is not True:
        return False
    if marker.get("source_name") != parquet_path.name:
        return False
    if marker.get("source_size") != parquet_path.stat().st_size:
        return False

    rows = marker.get("rows")
    written = marker.get("written")
    skipped = marker.get("skipped")
    if not isinstance(rows, int) or not isinstance(written, int) or not isinstance(skipped, int):
        return False
    if rows < 0 or written < 0 or skipped < 0:
        return False
    if written + skipped != rows:
        return False

    return True


def write_marker_atomic(marker_path: Path, payload: Dict) -> None:
    tmp = marker_path.parent / f".{marker_path.name}.marker_tmp"
    if tmp.exists():
        tmp.unlink()
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, marker_path)


def extract_audio_bytes(audio_obj) -> bytes:
    if audio_obj is None:
        raise ValueError("audio column is None")

    if hasattr(audio_obj, "as_py"):
        return extract_audio_bytes(audio_obj.as_py())

    if isinstance(audio_obj, dict):
        if "bytes" in audio_obj and audio_obj["bytes"] is not None:
            value = audio_obj["bytes"]
            if isinstance(value, bytes):
                return value
            if isinstance(value, bytearray):
                return bytes(value)
            if isinstance(value, memoryview):
                return value.tobytes()
        if "path" in audio_obj and audio_obj["path"]:
            path = Path(str(audio_obj["path"]))
            if path.exists():
                return path.read_bytes()
        raise ValueError(f"Unsupported audio dict structure: {list(audio_obj.keys())}")

    if isinstance(audio_obj, bytes):
        return audio_obj
    if isinstance(audio_obj, bytearray):
        return bytes(audio_obj)
    if isinstance(audio_obj, memoryview):
        return audio_obj.tobytes()

    raise TypeError(f"Unsupported audio payload type: {type(audio_obj)!r}")


def normalize_wav_relpath(name: str) -> Path:
    raw = str(name).strip().replace("\\", "/")
    if not raw:
        raise ValueError("Empty file name")

    parts = [part for part in raw.split("/") if part not in ("", ".")]
    if not parts:
        raise ValueError("Empty file name")

    if any(part == ".." for part in parts):
        raise ValueError(f"Unsafe relative path in file_name: {name}")

    # RMIS file_name values are often rooted at "audio/...". The extracted layout
    # should be subset_dir/<rest>, not subset_dir/audio/<rest>.
    if parts and parts[0] == "audio":
        parts = parts[1:]

    if not parts:
        raise ValueError(f"Invalid file_name after stripping root prefix: {name}")

    relpath = Path(*parts)
    if relpath.suffix == "":
        relpath = relpath.with_suffix(".wav")
    return relpath


def derive_wav_path(
    subset_dir: Path,
    file_name_value,
    audio_obj,
    parquet_stem: str,
    row_index: int,
    audio_bytes: bytes,
    force_reextract: bool,
) -> Tuple[Path, bool]:
    relpath: Optional[Path] = None

    if file_name_value is not None:
        value = str(file_name_value).strip()
        if value:
            relpath = normalize_wav_relpath(value)

    if relpath is None and isinstance(audio_obj, dict) and audio_obj.get("path"):
        relpath = normalize_wav_relpath(str(audio_obj["path"]))

    if relpath is None:
        relpath = Path(f"{parquet_stem}__{row_index:08d}.wav")

    candidate = subset_dir / relpath
    if candidate.exists() and not force_reextract:
        if candidate.stat().st_size == len(audio_bytes):
            return candidate, True
        alt = candidate.with_name(
            f"{candidate.stem}__{parquet_stem}_{row_index:08d}{candidate.suffix}"
        )
        if alt.exists() and alt.stat().st_size == len(audio_bytes):
            return alt, True
        return alt, False

    return candidate, False


def write_bytes_atomic(dst: Path, data: bytes) -> None:
    ensure_dir(dst.parent)
    tmp = dst.parent / f".{dst.name}.writing"
    if tmp.exists():
        tmp.unlink()
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, dst)


def extract_one_parquet(parquet_path: Path, subset_dir: Path, force_reextract: bool) -> Tuple[Dict[str, int], Path]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "pyarrow is required for parquet extraction. Please install it with "
            "`pip install pyarrow`."
        ) from exc

    marker_path = subset_dir / f".{parquet_path.name}.extract.ok.json"
    if not force_reextract:
        marker = load_marker(marker_path)
        if marker is not None and marker_matches_source(marker, parquet_path):
            print(f"[SKIP] {parquet_path.name} already extracted")
            return (
                {
                    "rows": int(marker.get("rows", 0)),
                    "written": int(marker.get("written", 0)),
                    "skipped": int(marker.get("skipped", 0)),
                },
                marker_path,
            )

    parquet_file = pq.ParquetFile(parquet_path)
    schema_names = parquet_file.schema_arrow.names
    if "audio" not in schema_names:
        raise KeyError(
            f"`audio` column not found in logical Arrow schema of {parquet_path}. "
            f"Available columns: {schema_names}"
        )
    columns = ["audio"]
    has_file_name = "file_name" in schema_names
    if has_file_name:
        columns.append("file_name")

    total_rows = 0
    written = 0
    skipped = 0

    for batch in parquet_file.iter_batches(columns=columns, use_threads=True):
        batch_size = batch.num_rows
        audio_col = batch.column(batch.schema.get_field_index("audio"))
        if has_file_name:
            file_name_col = batch.column(batch.schema.get_field_index("file_name"))
        else:
            file_name_col = None

        for idx in range(batch_size):
            global_row_index = total_rows + idx
            audio_obj = audio_col[idx].as_py()
            audio_bytes = extract_audio_bytes(audio_obj)
            file_name_value = file_name_col[idx].as_py() if file_name_col is not None else None

            wav_path, can_skip = derive_wav_path(
                subset_dir=subset_dir,
                file_name_value=file_name_value,
                audio_obj=audio_obj,
                parquet_stem=parquet_path.stem,
                row_index=global_row_index,
                audio_bytes=audio_bytes,
                force_reextract=force_reextract,
            )

            if can_skip and not force_reextract:
                skipped += 1
                continue

            write_bytes_atomic(wav_path, audio_bytes)
            written += 1

        total_rows += batch_size

    marker_payload = {
        "completed": True,
        "source_name": parquet_path.name,
        "source_size": parquet_path.stat().st_size,
        "rows": total_rows,
        "written": written,
        "skipped": skipped,
    }
    write_marker_atomic(marker_path, marker_payload)

    return (
        {"rows": total_rows, "written": written, "skipped": skipped},
        marker_path,
    )


def process_subset(
    snapshot_root: Path,
    output_dir: Path,
    subset: str,
    force_redownload: bool,
    force_reextract: bool,
    remove_parquet_after_extract: bool,
) -> None:
    subset_dir = output_dir / subset
    parquets = sync_subset_parquets(
        snapshot_root=snapshot_root,
        output_dir=output_dir,
        subset=subset,
        force_redownload=force_redownload,
    )

    subset_written = 0
    subset_skipped = 0
    subset_rows = 0

    for parquet_path in parquets:
        stats, marker_path = extract_one_parquet(
            parquet_path=parquet_path,
            subset_dir=subset_dir,
            force_reextract=force_reextract,
        )
        subset_rows += stats["rows"]
        subset_written += stats["written"]
        subset_skipped += stats["skipped"]

        if remove_parquet_after_extract:
            if parquet_path.exists():
                parquet_path.unlink()
                print(f"[CLEAN] Removed {parquet_path.name}")
            if marker_path.exists():
                marker_path.unlink()
                print(f"[CLEAN] Removed {marker_path.name}")

    print(
        f"[DONE] {subset}: rows={subset_rows}, written={subset_written}, "
        f"skipped_existing={subset_skipped}"
    )


def main() -> int:
    args = parse_args()
    subsets = validate_subsets(args.subset or DEFAULT_SUBSETS)

    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_dir(output_dir)

    token = args.token if args.token is not None else os.getenv("HF_TOKEN")
    allow_patterns = [f"{subset}/*.parquet" for subset in subsets]

    print(f"[INFO] Repo: {args.repo_id}")
    print(f"[INFO] Subsets: {', '.join(subsets)}")
    print(f"[INFO] Output: {output_dir}")

    snapshot_path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        cache_dir=args.cache_dir,
        token=token,
        allow_patterns=allow_patterns,
    )
    snapshot_root = Path(snapshot_path)
    print(f"[INFO] Snapshot ready at: {snapshot_root}")

    for subset in subsets:
        print(f"\n[SUBSET] {subset}")
        process_subset(
            snapshot_root=snapshot_root,
            output_dir=output_dir,
            subset=subset,
            force_redownload=args.force_redownload,
            force_reextract=args.force_reextract,
            remove_parquet_after_extract=args.remove_parquet_after_extract,
        )

    print("\n[ALL DONE]")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[ABORTED] Interrupted by user.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
