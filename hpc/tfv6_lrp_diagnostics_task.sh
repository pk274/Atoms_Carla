#!/bin/bash -l
# tfv6_lrp_diagnostics_task.sh
# Single-node SLURM job: runs the TFV6 LRP diagnostic suite (D01-D13) against
# a real checkpoint + real frames. Not an array job -- one job, one report.
#
# Variables injected by submit_tfv6_lrp_diagnostics.sh via --export:
#   CODE_DIR     project root (used for PYTHONPATH)
#   FRAMES_DIR   directory of run_*.npz frame files
#   OUT_DIR      output directory for the report + per-frame arrays
#   N_RUNS       number of run files to load (default 2)
#   N_FRAMES     number of frames to sample from the loaded runs (default 8)
#
# Optional rule-composite overrides (export before calling submit_*, e.g.
# `UITB=1 bash hpc/submit_tfv6_lrp_diagnostics.sh ...`, same convention as
# PGD_EPSILON for submit_test.sh) -- see LRPTFv6Model docstring for what
# each does:
#   UITB         1 -> AlphaBeta(2,1) instead of (1,0) for Conv/FFN
#   ZERO_BIAS    1 -> exclude bias from the AlphaBeta denominator (AnyLinear only)

#SBATCH -J tfv6_lrp_diag
#SBATCH -o /ptmp/%u/tfv6_lrp_diag/logs/diag_%j.out
#SBATCH -e /ptmp/%u/tfv6_lrp_diag/logs/diag_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24000MB
#SBATCH --time=02:00:00
# Add your account/partition here if required by your allocation, e.g.:
# #SBATCH --account=YOUR_ACCOUNT

module purge
module load python-waterboa/2025.06
source /u/$USER/venvs/pcla/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# hpc/stubs must come first so its carla.py/beartype.py stubs shadow any real
# install; CODE_DIR for ATOMs_Analysis.*; pcla_agents/transfuserv6 for lead.*
export PYTHONPATH="$CODE_DIR/hpc/stubs:$CODE_DIR:$CODE_DIR/pcla_agents/transfuserv6:$PYTHONPATH"

mkdir -p "$OUT_DIR" /ptmp/$USER/tfv6_lrp_diag/logs

EXTRA_FLAGS=()
[ "${UITB:-0}" = "1" ]      && EXTRA_FLAGS+=(--uitb)
[ "${ZERO_BIAS:-0}" = "1" ] && EXTRA_FLAGS+=(--zero-bias)

echo "=== TFV6 LRP Diagnostics (D01-D13) ==="
echo "Frames dir : $FRAMES_DIR"
echo "Out dir    : $OUT_DIR"
echo "N_RUNS     : ${N_RUNS:-2}"
echo "N_FRAMES   : ${N_FRAMES:-8}"
echo "UITB       : ${UITB:-0}"
echo "ZERO_BIAS  : ${ZERO_BIAS:-0}"
echo "Node       : $(hostname)"
echo "CPUs       : $SLURM_CPUS_PER_TASK"
date

srun python3 "$CODE_DIR/ATOMs_Analysis/utils/tfv6_lrp_diagnostics.py" \
    --frames-dir "$FRAMES_DIR" \
    --n-runs     "${N_RUNS:-2}" \
    --n-frames   "${N_FRAMES:-8}" \
    --out-dir    "$OUT_DIR" \
    "${EXTRA_FLAGS[@]}"

echo "Diagnostics finished with exit code $?"
echo "Report: $OUT_DIR/tfv6_diagnostics_report.txt"
date
