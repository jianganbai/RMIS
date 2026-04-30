#!/usr/bin/env bash
set -u
set -o pipefail

export LANG=C.UTF-8
export LC_ALL=C.UTF-8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DATASET_DIR="${BASE_DIR}/datasets_raw/sdust"
ALT_DATASET_DIR="${BASE_DIR}/datasets_raw/SDUST-Dataset-main"
LOG_DIR="${SCRIPT_DIR}/logs"
LOG_FILE="${LOG_DIR}/$(date +"%Y-%m-%d_%H-%M-%S")_processing_log.txt"

mkdir -p "$LOG_DIR"

timestamp() {
    date +"%Y-%m-%d %H:%M:%S"
}

log_info() {
    local message="$*"
    printf "[%s] [INFO] %s\n" "$(timestamp)" "$message" | tee -a "$LOG_FILE"
}

log_error() {
    local message="$*"
    printf "[%s] [ERROR] %s\n" "$(timestamp)" "$message" | tee -a "$LOG_FILE" >&2
}

ensure_dataset_dir() {
    if [[ -d "$DATASET_DIR" ]]; then
        return 0
    fi
    if [[ -d "$ALT_DATASET_DIR" ]]; then
        log_info "Renaming $ALT_DATASET_DIR to $DATASET_DIR"
        if ! mv "$ALT_DATASET_DIR" "$DATASET_DIR"; then
            log_error "Failed to rename $ALT_DATASET_DIR to $DATASET_DIR"
            exit 1
        fi
        return 0
    fi
    log_error "Dataset directory not found: $DATASET_DIR"
    log_error "Alternative directory not found: $ALT_DATASET_DIR"
    exit 1
}

log_info "Starting SDUST gear preprocessing."
log_info "Dataset directory: $DATASET_DIR"

ensure_dataset_dir

gear_cn="齿轮数据集"
gear_dir="${DATASET_DIR}/gear"
gear_cn_dir="${DATASET_DIR}/${gear_cn}"

if [[ -d "$gear_dir" ]]; then
    log_info "Found gear directory: $gear_dir"
elif [[ -d "$gear_cn_dir" ]]; then
    log_info "Renaming $gear_cn_dir to $gear_dir"
    if ! mv "$gear_cn_dir" "$gear_dir"; then
        log_error "Failed to rename $gear_cn_dir to $gear_dir"
        exit 1
    fi
else
    log_error "Expected directory not found: $gear_cn_dir"
    exit 1
fi

if [[ ! -d "${gear_dir}/NC" ]]; then
    log_error "Expected directory not found: ${gear_dir}/NC"
    exit 1
fi

rename_pairs=(
    "太阳断裂:sunfracture"
    "太阳点蚀:sunpitting"
    "太阳磨损:sunwear"
    "行星断裂:planetrayfracture"
    "行星点蚀:planetraypitting"
    "行星磨损:planetraywear"
)

for pair in "${rename_pairs[@]}"; do
    src_name="${pair%%:*}"
    dst_name="${pair#*:}"
    src_path="${gear_dir}/${src_name}"
    dst_path="${gear_dir}/${dst_name}"

    if [[ -d "$dst_path" ]]; then
        continue
    fi
    if [[ -d "$src_path" ]]; then
        log_info "Renaming $src_path to $dst_path"
        if ! mv "$src_path" "$dst_path"; then
            log_error "Failed to rename $src_path to $dst_path"
            exit 1
        fi
    else
        log_error "Expected directory not found: $src_path"
        exit 1
    fi
done

if ! command -v python >/dev/null 2>&1; then
    log_error "Python is not available in PATH."
    exit 1
fi

temp_dir="${BASE_DIR}/datasets_temp/sdust_gear"
wav_dir="${BASE_DIR}/datasets_wav/sdust_gear"

log_info "Running sdust_gear_mat2wav_20s.py"
if python "${SCRIPT_DIR}/sdust_gear_mat2wav_20s.py" --input_dir "$gear_dir" --output_dir "$temp_dir" 2>&1 | tee -a "$LOG_FILE"; then
    log_info "MAT to WAV conversion completed."
else
    log_error "MAT to WAV conversion failed."
    exit 1
fi

log_info "Running sdust_gear_split_10.py"
if python "${SCRIPT_DIR}/sdust_gear_split_10.py" --in_dir "$temp_dir" --out_dir "$wav_dir" 2>&1 | tee -a "$LOG_FILE"; then
    log_info "Split completed."
else
    log_error "Split failed."
    exit 1
fi

log_info "SDUST gear preprocessing completed."
