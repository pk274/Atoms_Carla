#!/bin/bash
# submit_tfv6_lrp_diagnostics.sh
# -------------------------------
# Submit a single SLURM job that runs the TFV6 LRP diagnostic suite
# (D01-D13, see ATOMs_Analysis/utils/tfv6_lrp_diagnostics.py) against a real
# checkpoint + real frames. Not an array job -- one job, one report.
#
# Usage (from $CODE_DIR on the HPC):
#   bash hpc/submit_tfv6_lrp_diagnostics.sh <FRAMES_DIR> [OUT_DIR] [N_RUNS] [N_FRAMES] [CODE_DIR]
#
# Arguments:
#   FRAMES_DIR  directory containing run_*.npz frame files
#               e.g. /ptmp/$USER/atoms_baseline/frames
#   OUT_DIR     where the report + per-frame .npy arrays are written
#               (default: /ptmp/$USER/tfv6_lrp_diag/out)
#   N_RUNS      number of run files to load (default: 2)
#   N_FRAMES    number of frames to sample from the loaded runs (default: 8)
#   CODE_DIR    project root (default: directory of this script's parent)
#
# Example:
#   bash hpc/submit_tfv6_lrp_diagnostics.sh /ptmp/$USER/atoms_baseline/frames

set -euo pipefail

FRAMES_DIR="${1:?Error: FRAMES_DIR not set. Usage: $0 <FRAMES_DIR> [OUT_DIR] [N_RUNS] [N_FRAMES] [CODE_DIR]}"
OUT_DIR="${2:-/ptmp/$USER/tfv6_lrp_diag/out}"
N_RUNS="${3:-2}"
N_FRAMES="${4:-8}"
CODE_DIR="${5:-$(cd "$(dirname "$0")/.." && pwd)}"

mkdir -p "/ptmp/$USER/tfv6_lrp_diag/logs" "$OUT_DIR"

echo "=== TFV6 LRP Diagnostics SLURM Submission ==="
echo "FRAMES_DIR : $FRAMES_DIR"
echo "OUT_DIR    : $OUT_DIR"
echo "N_RUNS     : $N_RUNS"
echo "N_FRAMES   : $N_FRAMES"
echo "CODE_DIR   : $CODE_DIR"
echo ""

JOB_ID=$(sbatch --parsable \
    --chdir="$CODE_DIR" \
    --export=ALL,CODE_DIR="$CODE_DIR",FRAMES_DIR="$FRAMES_DIR",OUT_DIR="$OUT_DIR",N_RUNS="$N_RUNS",N_FRAMES="$N_FRAMES" \
    "$CODE_DIR/hpc/tfv6_lrp_diagnostics_task.sh")

echo "Submitted job: $JOB_ID"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f /ptmp/\$USER/tfv6_lrp_diag/logs/diag_${JOB_ID}.out"
echo ""
echo "Report + per-frame arrays will be written to: $OUT_DIR"
