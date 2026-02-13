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

log_info "Starting SDUST bearing preprocessing."
log_info "Dataset directory: $DATASET_DIR"

ensure_dataset_dir

bearing_cn="轴承数据集"
bearing_dir="${DATASET_DIR}/bearing"
bearing_cn_dir="${DATASET_DIR}/${bearing_cn}"

if [[ -d "$bearing_dir" ]]; then
    log_info "Found bearing directory: $bearing_dir"
elif [[ -d "$bearing_cn_dir" ]]; then
    log_info "Renaming $bearing_cn_dir to $bearing_dir"
    if ! mv "$bearing_cn_dir" "$bearing_dir"; then
        log_error "Failed to rename $bearing_cn_dir to $bearing_dir"
        exit 1
    fi
else
    log_error "Expected directory not found: $bearing_cn_dir"
    exit 1
fi

if ! command -v python >/dev/null 2>&1; then
    log_error "Python is not available in PATH."
    exit 1
fi

temp_dir="${BASE_DIR}/datasets_temp/sdust_bearing"
wav_dir="${BASE_DIR}/datasets_wav/sdust_bearing"

log_info "Running sdust_bearing_mat2wav_40s.py"
if python "${SCRIPT_DIR}/sdust_bearing_mat2wav_40s.py" --input_dir "$bearing_dir" --output_dir "$temp_dir" 2>&1 | tee -a "$LOG_FILE"; then
    log_info "MAT to WAV conversion completed."
else
    log_error "MAT to WAV conversion failed."
    exit 1
fi

log_info "Running sdust_bearing_split_10.py"
if python "${SCRIPT_DIR}/sdust_bearing_split_10.py" --in_dir "$temp_dir" --out_dir "$wav_dir" 2>&1 | tee -a "$LOG_FILE"; then
    log_info "Split completed."
else
    log_error "Split failed."
    exit 1
fi

log_info "SDUST bearing preprocessing completed."
