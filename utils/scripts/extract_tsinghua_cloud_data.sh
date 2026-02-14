#!/usr/bin/env bash

# ==============================================================================
# RMIS Tsinghua Cloud Data Processing Tool
#
# This script is designed specifically for RMIS benchmark data downloaded from
# Tsinghua Cloud. It expects the exact file structure and checksum format used
# by the Tsinghua Cloud release and will not work correctly with data downloaded
# directly from original source websites.
# ==============================================================================

set -u
set -o pipefail

DATASET_DIR=$1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs/tsinghua_cloud"
LOG_FILE="${LOG_DIR}/$(date +"%Y-%m-%d_%H-%M-%S")_processing_log.txt"

mkdir -p "$LOG_DIR"

timestamp() {
    date +"%Y-%m-%d %H:%M:%S"
}

log_info() {
    local message="$*"
    printf "[%s] [INFO] %s\n" "$(timestamp)" "$message" | tee -a "$LOG_FILE"
}

log_warn() {
    local message="$*"
    printf "[%s] [WARN] %s\n" "$(timestamp)" "$message" | tee -a "$LOG_FILE"
}

log_error() {
    local message="$*"
    printf "[%s] [ERROR] %s\n" "$(timestamp)" "$message" | tee -a "$LOG_FILE" >&2
}

select_checksum_file() {
    local candidates=(
        "${DATASET_DIR}/check_sums.md5"
        "${SCRIPT_DIR}/check_sums.md5"
        "$(pwd)/check_sums.md5"
    )
    for candidate in "${candidates[@]}"; do
        if [[ -f "$candidate" ]]; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}

md5_command() {
    if command -v md5sum >/dev/null 2>&1; then
        echo "md5sum"
    elif command -v md5 >/dev/null 2>&1; then
        echo "md5"
    elif command -v openssl >/dev/null 2>&1; then
        echo "openssl"
    else
        echo ""
    fi
}

compute_md5() {
    local file_path="$1"
    local cmd="$2"
    case "$cmd" in
        md5sum)
            md5sum "$file_path" | awk '{print $1}'
            ;;
        md5)
            md5 -q "$file_path"
            ;;
        openssl)
            openssl md5 "$file_path" | awk '{print $2}'
            ;;
        *)
            return 1
            ;;
    esac
}

resolve_path() {
    local raw_path="$1"
    local cleaned_path="${raw_path#\*}"
    if [[ "$cleaned_path" =~ ^[A-Za-z]:\\ || "$cleaned_path" =~ ^/ ]]; then
        echo "$cleaned_path"
        return 0
    fi
    echo "${DATASET_DIR}/${cleaned_path}"
}

extract_archive() {
    local archive_path="$1"
    local dest_dir
    dest_dir="$(dirname "$archive_path")"
    case "$archive_path" in
            *.zip.001)
            local base_name
            base_name="$(basename "$archive_path")"
            local dataset_name="${base_name%.zip.001}"
            if ! command -v 7z >/dev/null 2>&1; then
                log_error "${dataset_name} extraction requires 7z, but 7z is not installed. Extraction aborted."
                return 1
            fi
            7z x "$archive_path" -o"$dest_dir" -mmt=on >/dev/null
            ;;
        *.zip.0[0-9][0-9]|*.zip.[0-9][0-9][0-9])
            log_info "Skipping split ZIP part: $archive_path (handled via .zip.001)"
            ;;
        *.tar.gz|*.tgz)
            tar -xzf "$archive_path" -C "$dest_dir"
            ;;
        *.tar.bz2|*.tbz2)
            tar -xjf "$archive_path" -C "$dest_dir"
            ;;
        *.tar.xz|*.txz)
            tar -xJf "$archive_path" -C "$dest_dir"
            ;;
        *.tar)
            tar -xf "$archive_path" -C "$dest_dir"
            ;;
        *.zip)
            if command -v 7z >/dev/null 2>&1; then
                7z x "$archive_path" -o"$dest_dir" -mmt=on >/dev/null 2>>"$LOG_FILE"
            else
                if ! command -v unzip >/dev/null 2>&1; then
                    log_error "Cannot extract $archive_path because neither '7z' nor 'unzip' is available."
                    return 1
                fi
                unzip -q "$archive_path" -d "$dest_dir"
            fi
            ;;
        *.7z)
            if ! command -v 7z >/dev/null 2>&1; then
                log_error "Cannot extract $archive_path because '7z' is not available."
                return 1
            fi
            7z x -y "$archive_path" -o"$dest_dir" >/dev/null
            ;;
        *.rar)
            if ! command -v unrar >/dev/null 2>&1; then
                log_error "Cannot extract $archive_path because 'unrar' is not available."
                return 1
            fi
            unrar x -o+ "$archive_path" "$dest_dir" >/dev/null
            ;;
        *.gz)
            if ! command -v gunzip >/dev/null 2>&1; then
                log_error "Cannot extract $archive_path because 'gunzip' is not available."
                return 1
            fi
            gunzip -k "$archive_path"
            ;;
        *.tar.gz.00|*.tar.gz.001)
            # Handle multi-volume tar.gz archives (part 00/001)
            local base_name="${archive_path%.*}"  # Remove .00/.001 extension
            
            # Use ls with sort to ensure correct physical order of parts (001, 002...)
            # We capture the output of ls and sort it explicitly
            local parts
            parts=$(ls "${base_name}".[0-9]* 2>/dev/null | sort)
            
            if [[ -z "$parts" ]]; then
                log_error "No parts found for multi-volume archive: $base_name"
                return 1
            fi

            log_info "Detected multi-volume archive start: $archive_path"
            # Format parts list for logging (replace newlines with spaces)
            local parts_log
            parts_log=$(echo "$parts" | tr '\n' ' ')
            log_info "Combining parts: $parts_log"
            
            # Use cat with the sorted parts list
            # Note: $parts contains newlines, unquoted usage relies on IFS to split into arguments for cat
            if cat $parts | tar -xz -C "$dest_dir"; then
                log_info "Successfully combined and extracted multi-volume archive: $base_name"
            else
                log_error "Failed to extract multi-volume archive: $base_name"
                return 1
            fi
            ;;
        *.tar.gz.[0-9][0-9]|*.tar.gz.[0-9][0-9][0-9])
            # Skip other parts of multi-volume archives as they are handled by the main part
            log_info "Skipping individual extraction of multi-volume part: $archive_path"
            ;;
        *)
            log_error "Unsupported archive format: $archive_path"
            return 1
            ;;
    esac
}

log_info "Starting RMIS Tsinghua Cloud data processing."
log_info "Script directory: $SCRIPT_DIR"
log_info "Expected datasets directory: $DATASET_DIR"

if [[ ! -d "$DATASET_DIR" ]]; then
    log_error "Datasets directory not found: $DATASET_DIR"
    exit 1
fi

CHECKSUM_FILE="$(select_checksum_file || true)"
if [[ -z "$CHECKSUM_FILE" ]]; then
    log_error "check_sums.md5 not found."
    log_error "Place check_sums.md5 inside $DATASET_DIR"
    exit 1
fi

MD5_TOOL="$(md5_command)"
if [[ -z "$MD5_TOOL" ]]; then
    log_error "No MD5 tool found. Install 'md5sum', 'md5', or 'openssl' to continue."
    exit 1
fi

log_info "Using checksum file: $CHECKSUM_FILE"
log_info "Using MD5 tool: $MD5_TOOL"

total_entries=0
missing_count=0
invalid_count=0
verified_count=0
extract_fail_count=0

missing_files=()
invalid_files=()
verified_archives=()

while IFS= read -r line || [[ -n "$line" ]]; do
    line="$(echo "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
    if [[ -z "$line" || "$line" == \#* ]]; then
        continue
    fi

    checksum=""
    file_path=""

    if [[ "$line" =~ ^MD5[[:space:]]*\((.+)\)[[:space:]]*=[[:space:]]*([A-Fa-f0-9]{32})$ ]]; then
        file_path="${BASH_REMATCH[1]}"
        checksum="${BASH_REMATCH[2]}"
    elif [[ "$line" =~ ^([A-Fa-f0-9]{32})[[:space:]]+(\*?)(.+)$ ]]; then
        checksum="${BASH_REMATCH[1]}"
        file_path="${BASH_REMATCH[3]}"
    else
        checksum="${line%% *}"
        file_path="${line#* }"
        # Trim leading whitespace if regex didn't match (fallback)
        file_path="${file_path#"${file_path%%[![:space:]]*}"}"
        file_path="${file_path#\*}"
    fi

    if [[ -z "$checksum" || -z "$file_path" || "$checksum" == "$line" ]]; then
        log_warn "Skipping unrecognized checksum line: $line"
        continue
    fi

    total_entries=$((total_entries + 1))
    resolved_path="$(resolve_path "$file_path")"

    if [[ ! -f "$resolved_path" ]]; then
        log_error "Missing archive referenced in checksum file: $resolved_path"
        missing_files+=("$resolved_path")
        missing_count=$((missing_count + 1))
        continue
    fi

    actual_checksum="$(compute_md5 "$resolved_path" "$MD5_TOOL" || true)"
    if [[ -z "$actual_checksum" ]]; then
        log_error "Failed to compute MD5 for: $resolved_path"
        invalid_files+=("$resolved_path")
        invalid_count=$((invalid_count + 1))
        continue
    fi

    if [[ "${actual_checksum,,}" != "${checksum,,}" ]]; then
        log_error "MD5 mismatch for $resolved_path (expected $checksum, got $actual_checksum)"
        invalid_files+=("$resolved_path")
        invalid_count=$((invalid_count + 1))
        continue
    fi

    log_info "MD5 verified: $resolved_path"
    verified_archives+=("$resolved_path")
    verified_count=$((verified_count + 1))
done < "$CHECKSUM_FILE"

if [[ $verified_count -eq 0 ]]; then
    log_warn "No valid archives found to extract."
else
    log_info "Beginning extraction of verified archives."
    for archive in "${verified_archives[@]}"; do
        log_info "Extracting archive: $archive"
        if extract_archive "$archive"; then
            log_info "Extraction complete: $archive"
        else
            log_error "Extraction failed for: $archive"
            extract_fail_count=$((extract_fail_count + 1))
        fi
    done
fi

log_info "Processing summary:"
log_info "Checksum entries processed: $total_entries"
log_info "Archives verified: $verified_count"
log_info "Missing archives: $missing_count"
log_info "Checksum failures: $invalid_count"
log_info "Extraction failures: $extract_fail_count"

if [[ $missing_count -gt 0 ]]; then
    log_warn "Missing archives list:"
    for missing in "${missing_files[@]}"; do
        log_warn " - $missing"
    done
fi

if [[ $invalid_count -gt 0 ]]; then
    log_warn "Checksum failure list:"
    for invalid in "${invalid_files[@]}"; do
        log_warn " - $invalid"
    done
fi

if [[ $extract_fail_count -gt 0 ]]; then
    log_error "One or more archives failed to extract. Review the log for details."
    exit 2
fi

log_info "RMIS Tsinghua Cloud data processing completed."
