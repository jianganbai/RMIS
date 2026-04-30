#!/usr/bin/env bash
set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DATASET_DIR="${BASE_DIR}/datasets_raw/umged"
ALT_DATASET_DIR="${BASE_DIR}/datasets_raw/GearEccDataset"
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

log_info "Starting UMGED preprocessing."
log_info "Dataset directory: $DATASET_DIR"

ensure_dataset_dir

required_dirs=("G1" "G2")
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

declare -A dataset_map=(
    ["sound"]="umged_sound"
    ["vibration"]="umged_vib"
    ["voltage"]="umged_vol"
    ["current"]="umged_cur"
)

for dataset_key in "sound" "vibration" "voltage" "current"; do
    output_name="${dataset_map[$dataset_key]}"
    temp_dir="${BASE_DIR}/datasets_temp/${output_name}"
    wav_dir="${BASE_DIR}/datasets_wav/${output_name}"

    log_info "Running umged_mat2wav_600s.py for ${dataset_key}"
    if python "${SCRIPT_DIR}/umged_mat2wav_600s.py" --input_dir "$DATASET_DIR" --output_dir "$temp_dir" --dataset "$dataset_key" 2>&1 | tee -a "$LOG_FILE"; then
        log_info "MAT to WAV conversion completed for ${dataset_key}"
    else
        log_error "MAT to WAV conversion failed for ${dataset_key}"
        exit 1
    fi

    log_info "Running umged_split_10.py for ${dataset_key}"
    if python "${SCRIPT_DIR}/umged_split_10.py" --in_dir "$temp_dir" --out_dir "$wav_dir" 2>&1 | tee -a "$LOG_FILE"; then
        log_info "Split completed for ${dataset_key}"
    else
        log_error "Split failed for ${dataset_key}"
        exit 1
    fi
done

log_info "UMGED preprocessing completed."
