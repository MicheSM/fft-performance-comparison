#!/usr/bin/env bash
set -euo pipefail

# Load shared environment variables
source "$(dirname "$0")/env.sh"

# Executable directory
EXEC_DIR="${BUILD_DIR}"

# Output directories
CT_OUT="${RAW_OUTPUT_DIR}/cooley-tuckey"
ST_OUT="${RAW_OUTPUT_DIR}/stockham"

mkdir -p "$CT_OUT" "$ST_OUT"

############ CLEAN OLD OUTPUTS ############
rm -f "${CT_OUT}"/* || true
rm -f "${ST_OUT}"/* || true

###########################################

# List of executables to run
executables=(
    fft_cooley_bi_novector
    fft_cooley_bi_auto
    fft_cooley_bi_sve
    fft_cooley_bi_sve_3loop
    fft_cooley_ci_novector
    fft_cooley_ci_auto
    fft_cooley_ci_sve
    fft_cooley_ci_sve_3loop
    fft_stockham_bi_novector
    fft_stockham_bi_auto
    fft_stockham_bi_sve
    fft_stockham_ci_novector
    fft_stockham_ci_auto
    fft_stockham_ci_sve
)

sizes=(
    8 16 32 64 128 256 512 1024 2048 4096 8192
    16384 32768 65536 131072 262144 524288 1048576
    2097152 4194304 8388608
)

###########################################

for exe in "${executables[@]}"; do
    EXE_PATH="${EXEC_DIR}/${exe}"

    if [[ ! -x "$EXE_PATH" ]]; then
        echo "❌ Executable not found: $EXE_PATH"
        continue
    fi

    # Decide output directory based on name
    if [[ "$exe" == fft_cooley* ]]; then
        OUTFILE="${CT_OUT}/output_${exe}.txt"
    else
        OUTFILE="${ST_OUT}/output_${exe}.txt"
    fi

    echo "▶ Running $exe → $OUTFILE"

    {
        for n in "${sizes[@]}"; do
            for r in {1..32}; do
                printf "%s " "$n"
                "$EXE_PATH" "$n"
            done
        done
    } > "$OUTFILE"

    echo "✔ Done: $exe"
done
