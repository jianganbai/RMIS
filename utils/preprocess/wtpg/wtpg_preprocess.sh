#!/usr/bin/env bash
set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DATASET_DIR="${BASE_DIR}/datasets_raw/wtpg"
ALT_DATASET_DIR="${BASE_DIR}/datasets_raw/WT-planetary-gearbox-dataset"
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

log_info "Starting WTPG preprocessing."
log_info "Dataset directory: $DATASET_DIR"

ensure_dataset_dir

required_dirs=("broken" "healthy" "missing_tooth" "root_crack" "wear")
for dir_name in "${required_dirs[@]}"; do
    if [[ ! -d "${DATASET_DIR}/${dir_name}" ]]; then
        log_error "Expected directory not found: ${DATASET_DIR}/${dir_name}"
        exit 1
    fi
done

if ! command -v python >/dev/null 2>&1; then
    log_error "Python is not available in PATH."
    exit 1
fi

temp_dir="${BASE_DIR}/datasets_temp/wtpg"
wav_dir="${BASE_DIR}/datasets_wav/wtpg"

log_info "Running wtpg_MAT2flac.py"
if python "${SCRIPT_DIR}/wtpg_MAT2flac.py" --input_dir "$DATASET_DIR" --output_dir "$temp_dir" 2>&1 | tee -a "$LOG_FILE"; then
    log_info "MAT to FLAC conversion completed."
else
    log_error "MAT to FLAC conversion failed."
    exit 1
fi

log_info "Running wtpg_split_10.py"
if python "${SCRIPT_DIR}/wtpg_split_10.py" --in_dir "$temp_dir" --out_dir "$wav_dir" 2>&1 | tee -a "$LOG_FILE"; then
    log_info "Split completed."
else
    log_error "Split failed."
    exit 1
fi

log_info "WTPG preprocessing completed."
