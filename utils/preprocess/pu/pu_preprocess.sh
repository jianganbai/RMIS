#!/usr/bin/env bash
set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DATASET_DIR="${BASE_DIR}/datasets_raw/pu"
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

if [[ ! -d "$DATASET_DIR" ]]; then
    log_error "Dataset directory not found: $DATASET_DIR"
    exit 1
fi

raw_dirs=(
    K001 K002 K003 K004 K005 K006
    KI01 KI03 KI05 KI07 KI08
    KB23 KB24 KI04 KI14 KI16 KI17 KI18 KI21
    KA01 KA03 KA05 KA06 KA07 KA08 KA09
    KA04 KA15 KA16 KA22 KA30 KB27
)

missing_raw_dirs=()
for dir_name in "${raw_dirs[@]}"; do
    if [[ ! -d "${DATASET_DIR}/${dir_name}" ]]; then
        missing_raw_dirs+=("$dir_name")
    fi
done

has_all_raw_dirs=false
if [[ "${#missing_raw_dirs[@]}" -eq 0 ]]; then
    has_all_raw_dirs=true
fi

organized_ready=false
if [[ -d "${DATASET_DIR}/healthy" && -d "${DATASET_DIR}/IR" && -d "${DATASET_DIR}/OR" ]]; then
    organized_ready=true
fi

extractor=""
if command -v unrar >/dev/null 2>&1; then
    extractor="unrar"
elif command -v 7z >/dev/null 2>&1; then
    extractor="7z"
fi

mapfile -t rar_files < <(find "$DATASET_DIR" -maxdepth 1 -type f -name "*.rar" | sort)
rar_count="${#rar_files[@]}"
if [[ "$has_all_raw_dirs" == true ]]; then
    log_info "All 32 raw folders found. Skipping extraction."
elif [[ "$rar_count" -eq 32 ]]; then
    if [[ -z "$extractor" ]]; then
        log_error "Cannot extract archives because neither 'unrar' nor '7z' is available."
        exit 1
    fi
    for rar_file in "${rar_files[@]}"; do
        log_info "Extracting $rar_file"
        if [[ "$extractor" == "unrar" ]]; then
            if ! unrar x -o+ "$rar_file" "$DATASET_DIR" >/dev/null; then
                log_error "Failed to extract $rar_file"
                exit 1
            fi
        else
            if ! 7z x -y "$rar_file" -o"$DATASET_DIR" >/dev/null; then
                log_error "Failed to extract $rar_file"
                exit 1
            fi
        fi
    done
else
    log_error "Expected 32 raw folders or 32 rar files in $DATASET_DIR."
    if [[ "${#missing_raw_dirs[@]}" -gt 0 ]]; then
        log_error "Missing raw folders: ${missing_raw_dirs[*]}"
    fi
    log_error "Found rar files: $rar_count"
    exit 1
fi

organize_script="${SCRIPT_DIR}/pu_supplementary/pu_organize_files.sh"
if [[ ! -f "$organize_script" ]]; then
    log_error "Organize script not found: $organize_script"
    exit 1
fi

log_info "Running pu_organize_files.sh"
if [[ "$organized_ready" == true ]]; then
    log_info "Organized directories already exist. Skipping pu_organize_files.sh"
else
    if ! (cd "$DATASET_DIR" && bash "$organize_script"); then
        log_error "Failed to run pu_organize_files.sh"
        exit 1
    fi
fi

replace_src="${SCRIPT_DIR}/pu_supplementary/mat_files_err/OR/artificial/KA08/N15_M01_F10_KA08_2_new.mat"
replace_dst="${DATASET_DIR}/OR/artificial/KA08/N15_M01_F10_KA08_2.mat"

if [[ ! -f "$replace_src" ]]; then
    log_error "Replacement source file not found: $replace_src"
    exit 1
fi

if [[ ! -f "$replace_dst" ]]; then
    log_error "Replacement target file not found: $replace_dst"
    exit 1
fi

log_info "Replacing $replace_dst with $replace_src"
if ! cp -f "$replace_src" "$replace_dst"; then
    log_error "Failed to replace $replace_dst"
    exit 1
fi

required_dirs=("healthy" "IR" "OR")
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

log_info "Running pu_mat2wav_4s.py for current dataset"
if python "${SCRIPT_DIR}/pu_mat2wav_4s.py" --input_dir "$DATASET_DIR" --output_dir "${BASE_DIR}/datasets_wav/pu_cur" --dataset current 2>&1 | tee -a "$LOG_FILE"; then
    log_info "Current dataset conversion completed."
else
    log_error "Current dataset conversion failed."
    exit 1
fi

log_info "Running pu_mat2wav_4s.py for vibration dataset"
if python "${SCRIPT_DIR}/pu_mat2wav_4s.py" --input_dir "$DATASET_DIR" --output_dir "${BASE_DIR}/datasets_wav/pu_vib" --dataset vibration 2>&1 | tee -a "$LOG_FILE"; then
    log_info "Vibration dataset conversion completed."
else
    log_error "Vibration dataset conversion failed."
    exit 1
fi

log_info "PU preprocessing completed."
