#!/usr/bin/env bash
set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DATASET_DIR="${BASE_DIR}/datasets_raw/iica"
OUTPUT_DIR="${BASE_DIR}/datasets_wav/iica"
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

log_info "Starting IICA preprocessing."
log_info "Dataset directory: $DATASET_DIR"
log_info "Output directory: $OUTPUT_DIR"

require_dir "$DATASET_DIR"

expected_dirs=("tubeleak" "ventleak" "ventlow")
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
    if [[ "$zip_count" -ne 3 ]]; then
        log_error "Expected 3 zip files in $DATASET_DIR, found $zip_count."
        exit 1
    fi

    for zip_file in "${zip_files[@]}"; do
        log_info "Extracting $zip_file"
        if ! unzip -q "$zip_file" -d "$DATASET_DIR"; then
            log_error "Failed to extract $zip_file"
            exit 1
        fi
    done
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

log_info "Running iica_split_10.py"
if python "${SCRIPT_DIR}/iica_split_10.py" --in_dir "$DATASET_DIR" --out_dir "$OUTPUT_DIR" 2>&1 | tee -a "$LOG_FILE"; then
    log_info "IICA preprocessing completed."
else
    log_error "IICA preprocessing failed."
    exit 1
fi
