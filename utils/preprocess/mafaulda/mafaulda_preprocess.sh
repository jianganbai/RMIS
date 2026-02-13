#!/usr/bin/env bash
set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DATASET_DIR="${BASE_DIR}/datasets_raw/mafaulda"
OUTPUT_SOUND_DIR="${BASE_DIR}/datasets_wav/mafaulda_sound"
OUTPUT_VIB_DIR="${BASE_DIR}/datasets_wav/mafaulda_vib"
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

require_dir() {
    local dir_path="$1"
    if [[ ! -d "$dir_path" ]]; then
        log_error "Directory not found: $dir_path"
        exit 1
    fi
}

log_info "Starting MaFaulDa preprocessing."
log_info "Dataset directory: $DATASET_DIR"
log_info "Output directories: $OUTPUT_SOUND_DIR, $OUTPUT_VIB_DIR"

require_dir "$DATASET_DIR"

expected_dirs=("horizontal-misalignment" "imbalance" "normal" "overhang" "underhang" "vertical-misalignment")
all_present=true
for dir_name in "${expected_dirs[@]}"; do
    if [[ ! -d "${DATASET_DIR}/${dir_name}" ]]; then
        all_present=false
        break
    fi
done

if [[ "$all_present" != true ]]; then
    if ! command -v unzip >/dev/null 2>&1; then
        log_error "Cannot extract archives because 'unzip' is not available."
        exit 1
    fi

    mapfile -t zip_files < <(find "$DATASET_DIR" -maxdepth 1 -type f -name "*.zip" | sort)
    zip_count="${#zip_files[@]}"
    if [[ "$zip_count" -ne 1 ]]; then
        log_error "Expected 1 zip file in $DATASET_DIR, found $zip_count."
        exit 1
    fi

    log_info "Extracting ${zip_files[0]}"
    if ! unzip -q "${zip_files[0]}" -d "$DATASET_DIR"; then
        log_error "Failed to extract ${zip_files[0]}"
        exit 1
    fi
fi

for dir_name in "${expected_dirs[@]}"; do
    if [[ ! -d "${DATASET_DIR}/${dir_name}" ]]; then
        log_error "Expected directory not found after extraction: ${DATASET_DIR}/${dir_name}"
        exit 1
    fi
done

if ! command -v python >/dev/null 2>&1; then
    log_error "Python is not available in PATH."
    exit 1
fi

log_info "Running mafaulda_csv2wav_5s.py for sound dataset"
if python "${SCRIPT_DIR}/mafaulda_csv2wav_5s.py" --input_dir "$DATASET_DIR" --output_dir "$OUTPUT_SOUND_DIR" --dataset sound 2>&1 | tee -a "$LOG_FILE"; then
    log_info "Sound dataset conversion completed."
else
    log_error "Sound dataset conversion failed."
    exit 1
fi

log_info "Running mafaulda_csv2wav_5s.py for vibration dataset"
if python "${SCRIPT_DIR}/mafaulda_csv2wav_5s.py" --input_dir "$DATASET_DIR" --output_dir "$OUTPUT_VIB_DIR" --dataset vibration 2>&1 | tee -a "$LOG_FILE"; then
    log_info "Vibration dataset conversion completed."
else
    log_error "Vibration dataset conversion failed."
    exit 1
fi

log_info "MaFaulDa preprocessing completed."
