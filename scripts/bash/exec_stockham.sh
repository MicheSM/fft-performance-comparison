#!/usr/bin/env bash
# -e exit on error
# -u treat unset variables as errors
# -o pipefail return exit code of last failed command in a pipeline
set -euo pipefail

# Load shared environment variables
source "$(dirname "$0")/env.sh"

# Output directory
OUT_DIR="${DATA_DIR}/results/raw-outputs/stockham"

mkdir -p "$OUT_DIR"

# clean old outputs
rm -f "${OUT_DIR}"/* || true

# List of executables to run
executables=(
    fft_stockham_bi_novector
    fft_stockham_bi_auto
    fft_stockham_bi_sve
    fft_stockham_bi_sve_2loop
    fft_stockham_ci_novector
    fft_stockham_ci_auto
    fft_stockham_ci_sve
    fft_stockham_ci_sve_2loop
)

sizes=(
    8 16 32 64 128 256 512 1024 2048 4096 8192
    16384 32768 65536 131072 262144 524288 1048576
    2097152 
)

###########################################
for exe in "${executables[@]}"; do
    EXE_PATH="${BUILD_DIR}/${exe}"

    if [[ ! -x "$EXE_PATH" ]]; then
        echo "Executable not found: $EXE_PATH"
        continue
    fi

    OUTFILE="${OUT_DIR}/output_${exe}.txt"

    echo "Running $exe, writing $OUTFILE"

    {
        for n in "${sizes[@]}"; do
            for r in {1..4}; do
                taskset -c 10 "$EXE_PATH" "$n"
            done
        done
    } > "$OUTFILE"

    echo "Done: $exe"
done
