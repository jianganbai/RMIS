#!/bin/bash

# ==============================================================================
#  A simple script to reorganize pu dataset folders into a structured hierarchy.
#
#  Usage:
#  1. Save this file as 'pu_organize_files.sh' in the SAME directory where your
#     32 folders (KA01, K001, KI01, etc.) are located.
#  2. Give it execute permissions:  chmod +x pu_organize_files.sh
#  3. Run the script:               ./pu_organize_files.sh
# ==============================================================================

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
# The name of the new parent directory that will be created.
TARGET_DIR="./"

# 1. Create the target directory structure.
# The '-p' flag creates parent directories as needed and doesn't complain if they exist.
echo "Creating target directory structure under ./${TARGET_DIR}/"
mkdir -p "${TARGET_DIR}/healthy"
mkdir -p "${TARGET_DIR}/IR/artificial"
mkdir -p "${TARGET_DIR}/IR/lifetime"
mkdir -p "${TARGET_DIR}/OR/artificial"
mkdir -p "${TARGET_DIR}/OR/lifetime"

# 2. Move the folders into their new locations.
# The '-v' flag makes the 'mv' command verbose, showing what is being moved.
# We wrap each move command in a 'for' loop to handle missing folders gracefully.

echo -e "\nMoving 'healthy' folders..."
for dir in K00{1..6}; do
    [ -d "$dir" ] && mv -v "$dir" "${TARGET_DIR}/healthy/"
done

echo -e "\nMoving 'IR/artificial' folders..."
for dir in KI01 KI03 KI05 KI07 KI08; do
    [ -d "$dir" ] && mv -v "$dir" "${TARGET_DIR}/IR/artificial/"
done

echo -e "\nMoving 'IR/lifetime' folders..."
for dir in KB23 KB24 KI04 KI14 KI16 KI17 KI18 KI21; do
    [ -d "$dir" ] && mv -v "$dir" "${TARGET_DIR}/IR/lifetime/"
done

echo -e "\nMoving 'OR/artificial' folders..."
for dir in KA01 KA03 KA05 KA06 KA07 KA08 KA09; do
    [ -d "$dir" ] && mv -v "$dir" "${TARGET_DIR}/OR/artificial/"
done

echo -e "\nMoving 'OR/lifetime' folders..."
for dir in KA04 KA15 KA16 KA22 KA30 KB27; do
    [ -d "$dir" ] && mv -v "$dir" "${TARGET_DIR}/OR/lifetime/"
done

# --- Completion ---
echo -e "\n✅ Reorganization complete!"
echo "All folders have been moved into the './${TARGET_DIR}/' directory structure."
