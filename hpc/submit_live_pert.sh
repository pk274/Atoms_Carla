#!/bin/bash
# submit_live_pert.sh
# -------------------
# Submit the full live-perturbation ATOMs pipeline as chained SLURM jobs,
# one 3-job chain per source frame file.
#
# For each run_{PERTURBATION}_live_pert_*.npz file in FRAMES_DIR, a separate
# chain is submitted:
#   Job 1 (prep)   : copy single file → <variant>/live_pert_concat.npz
#   Job 2 (array)  : parallel LRP + ATOMs, one task per CHUNK_SIZE frames
#   Job 3 (gather) : concatenate partial profiles → <variant>/live_pert_profiles_<mode>.npy
#
# The variant name is derived from the filename by stripping the
# "run_{PERTURBATION}_live_pert_" prefix, e.g.:
#   run_pgd_live_pert_brake_205328_000.npz  →  variant: brake_205328_000
#
# Usage (from $CODE_DIR on the HPC):
#   bash hpc/submit_live_pert.sh <FRAMES_DIR> <WORK_DIR> <MODEL_DIR> <PERTURBATION> [CODE_DIR] [CHUNK_SIZE] [MODE_ANALYSIS]
#
# Arguments:
#   FRAMES_DIR     directory containing run_{PERTURBATION}_live_pert_*.npz files
#   WORK_DIR       working directory; per-variant subdirs are created here
#   MODEL_DIR      path to TFV6 pretrained model directory
#   PERTURBATION   perturbation name, e.g. "pgd"
#   CODE_DIR       project root (default: parent of this script)
#   CHUNK_SIZE     frames per array task (default: 20)
#   MODE_ANALYSIS  1 or 2 (default: 1)
#
# After all jobs finish, collect results with:
#   bash hpc/collect_results.sh live_pert tfv6 <MODE_ANALYSIS> <PERTURBATION>

set -euo pipefail

FRAMES_DIR="${1:?Error: FRAMES_DIR not set. Usage: $0 <FRAMES_DIR> <WORK_DIR> <MODEL_DIR> <PERTURBATION> [CODE_DIR] [CHUNK_SIZE] [MODE_ANALYSIS]}"
WORK_DIR="${2:?Error: WORK_DIR not set.}"
MODEL_DIR="${3:?Error: MODEL_DIR not set.}"
PERTURBATION="${4:?Error: PERTURBATION not set (e.g. 'pgd').}"
CODE_DIR="${5:-$(cd "$(dirname "$0")/.." && pwd)}"
CHUNK_SIZE="${6:-20}"
MODE_ANALYSIS="${7:-1}"

# Upper bound on frame count per file — tasks past the actual data exit cleanly.
MAX_FRAMES=200
N_TASKS=$(( (MAX_FRAMES + CHUNK_SIZE - 1) / CHUNK_SIZE ))
N_LAST=$(( N_TASKS - 1 ))

echo "=== ATOMs Live-Perturbation SLURM Submission ==="
echo "FRAMES_DIR    : $FRAMES_DIR"
echo "WORK_DIR      : $WORK_DIR"
echo "MODEL_DIR     : $MODEL_DIR"
echo "PERTURBATION  : $PERTURBATION"
echo "CODE_DIR      : $CODE_DIR"
echo "CHUNK_SIZE    : $CHUNK_SIZE"
echo "MODE_ANALYSIS : $MODE_ANALYSIS"
echo "N_TASKS/file  : $N_TASKS (indices 0–$N_LAST)"
echo ""

# --- Discover source files ---
mapfile -t PERT_FILES < <(ls "$FRAMES_DIR"/run_${PERTURBATION}_live_pert_*.npz 2>/dev/null | sort)
if [ ${#PERT_FILES[@]} -eq 0 ]; then
    echo "ERROR: No run_${PERTURBATION}_live_pert_*.npz files found in $FRAMES_DIR" >&2
    exit 1
fi
echo "Found ${#PERT_FILES[@]} file(s):"
printf '  %s\n' "${PERT_FILES[@]##*/}"
echo ""

# --- Submit one 3-job chain per file ---
for FPATH in "${PERT_FILES[@]}"; do
    FSTEM=$(basename "$FPATH" .npz)
    VARIANT="${FSTEM#run_${PERTURBATION}_live_pert_}"
    FILE_WORK="$WORK_DIR/$VARIANT"
    CONCAT_FILE="$FILE_WORK/live_pert_concat.npz"
    PARTIALS_DIR="$FILE_WORK/partials/mode_${MODE_ANALYSIS}"
    PROFILES_OUT="$FILE_WORK/live_pert_profiles_${MODE_ANALYSIS}.npy"
    LOG_DIR="$FILE_WORK/logs"

    mkdir -p "$LOG_DIR" "$PARTIALS_DIR"

    echo "=== Variant: $VARIANT ==="

    ARRAY_DEP=""
    if [ -f "$CONCAT_FILE" ]; then
        echo "  live_pert_concat.npz already exists — skipping prep job."
    else
        PREP_JOB_ID=$(sbatch --parsable \
            --output="$LOG_DIR/prep_%j.out" --error="$LOG_DIR/prep_%j.err" \
            --chdir="$CODE_DIR" \
            --export=ALL,FRAMES_DIR="$FRAMES_DIR",PERTURBATION="$PERTURBATION",CONCAT_FILE="$CONCAT_FILE",CODE_DIR="$CODE_DIR",FILE_PATH="$FPATH" \
            "$CODE_DIR/hpc/prep_live_pert_task.sh")
        echo "  Submitted prep  : $PREP_JOB_ID"
        ARRAY_DEP="--dependency=afterok:${PREP_JOB_ID}"
    fi

    ARRAY_JOB_ID=$(sbatch --parsable \
        --array=0-${N_LAST} \
        ${ARRAY_DEP} \
        --output="$LOG_DIR/chunk_%A_%a.out" --error="$LOG_DIR/chunk_%A_%a.err" \
        --chdir="$CODE_DIR" \
        --export=ALL,CONCAT_FILE="$CONCAT_FILE",PARTIALS_DIR="$PARTIALS_DIR",MODEL_DIR="$MODEL_DIR",CODE_DIR="$CODE_DIR",CHUNK_SIZE="$CHUNK_SIZE",MODE_ANALYSIS="$MODE_ANALYSIS" \
        "$CODE_DIR/hpc/array_live_pert_task.sh")
    echo "  Submitted array : $ARRAY_JOB_ID  ($N_TASKS tasks, indices 0–$N_LAST)"

    GATHER_JOB_ID=$(sbatch --parsable \
        --dependency=afterok:${ARRAY_JOB_ID} \
        --output="$LOG_DIR/gather_%j.out" --error="$LOG_DIR/gather_%j.err" \
        --chdir="$CODE_DIR" \
        --export=ALL,PARTIALS_DIR="$PARTIALS_DIR",PROFILES_OUT="$PROFILES_OUT",CODE_DIR="$CODE_DIR",MODE_ANALYSIS="$MODE_ANALYSIS" \
        "$CODE_DIR/hpc/gather_live_pert_task.sh")
    echo "  Submitted gather: $GATHER_JOB_ID"
    echo ""
done

echo "Monitor with:"
echo "  squeue -u \$USER"
echo ""
echo "After all gather jobs complete, collect results into the repo:"
echo "  cd /u/\$USER/pcla"
echo "  bash hpc/collect_results.sh live_pert tfv6 ${MODE_ANALYSIS} $PERTURBATION"
echo "  git commit -m 'add TFV6 live_pert $PERTURBATION results from HPC' && git push"
echo "Then locally: git pull, set RECOMPUTE_TEST_ATOMS=False in atoms_config.py"
