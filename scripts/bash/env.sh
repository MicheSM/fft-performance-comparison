#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}/../.."

BUILD_DIR="${ROOT_DIR}/build"
INPUT_DIR="${ROOT_DIR}/data/inputs"
RESULTS_DIR="${ROOT_DIR}/data/results"
RAW_OUTPUT_DIR="${RESULTS_DIR}/raw-outputs"
