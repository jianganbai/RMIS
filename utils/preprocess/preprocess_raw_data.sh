#!/usr/bin/env bash
set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs/raw_data"
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

dataset_scripts=(
    "iica:${SCRIPT_DIR}/iica/iica_preprocess.sh"
    "mafaulda:${SCRIPT_DIR}/mafaulda/mafaulda_preprocess.sh"
    "pu:${SCRIPT_DIR}/pu/pu_preprocess.sh"
    "sdust_bearing:${SCRIPT_DIR}/sdust_bearing/sdust_bearing_preprocess.sh"
    "sdust_gear:${SCRIPT_DIR}/sdust_gear/sdust_gear_preprocess.sh"
    "umged:${SCRIPT_DIR}/umged/umged_preprocess.sh"
    "wtpg:${SCRIPT_DIR}/wtpg/wtpg_preprocess.sh"
)

declare -A dataset_map
for item in "${dataset_scripts[@]}"; do
    name="${item%%:*}"
    path="${item#*:}"
    dataset_map["$name"]="$path"
done

selected_datasets=()
if [[ "$#" -eq 0 ]]; then
    selected_datasets=("${!dataset_map[@]}")
else
    for name in "$@"; do
        if [[ -n "${dataset_map[$name]:-}" ]]; then
            selected_datasets+=("$name")
        else
            log_error "Unknown dataset name: $name"
        fi
    done
fi

if [[ "${#selected_datasets[@]}" -eq 0 ]]; then
    log_error "No valid datasets selected."
    exit 1
fi

log_info "Starting raw data preprocessing."
for name in "${selected_datasets[@]}"; do
    script_path="${dataset_map[$name]}"
    if [[ ! -f "$script_path" ]]; then
        log_error "Script not found for ${name}: $script_path"
        continue
    fi
    dataset_log="${LOG_DIR}/$(date +"%Y-%m-%d_%H-%M-%S")_${name}.log"
    log_info "Running ${name} preprocessing."
    if bash "$script_path" 2>&1 | tee -a "$dataset_log"; then
        log_info "${name} preprocessing completed."
    else
        log_error "${name} preprocessing failed."
    fi
done

log_info "Raw data preprocessing completed."
