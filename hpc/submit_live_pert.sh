#!/bin/bash
# submit_live_pert.sh
# -------------------
# Submit the full live-perturbation ATOMs pipeline as three chained SLURM jobs:
#
#   Job 1 (prep)   : concatenate live_pert run files → live_pert_concat.npz
#   Job 2 (array)  : parallel LRP + ATOMs, one task per CHUNK_SIZE frames
#   Job 3 (gather) : concatenate partial profiles → live_pert_profiles.npy
#
# The live-pert data is already recorded with perturbations applied in CARLA,
# so no offline perturbation step is needed (unlike the test-set pipeline).
#
# Usage (from $CODE_DIR on the HPC):
#   bash hpc/submit_live_pert.sh <FRAMES_DIR> <WORK_DIR> <MODEL_DIR> <PERTURBATION> [CODE_DIR] [CHUNK_SIZE]
#
# Arguments:
#   FRAMES_DIR     directory containing run_{PERTURBATION}_live_pert_*.npz files
#                  e.g. /ptmp/$USER/atoms_live_pert/frames
#   WORK_DIR       working directory for all outputs (concat file, partials, logs)
#                  e.g. /ptmp/$USER/atoms_live_pert
#   MODEL_DIR      path to TFV6 pretrained model directory
#                  e.g. /u/$USER/pcla/pcla_agents/transfuserv6_pretrained/visiononly_resnet34
#   PERTURBATION   perturbation name, e.g. "pgd" — must match the recorded filenames
#   CODE_DIR       project root (default: parent of this script)
#   CHUNK_SIZE     frames per array task (default: 20; 10 tasks for 200 frames)
#
# Example:
#   bash hpc/submit_live_pert.sh \
#       /ptmp/$USER/atoms_live_pert/frames \
#       /ptmp/$USER/atoms_live_pert \
#       /u/$USER/pcla/pcla_agents/transfuserv6_pretrained/visiononly_resnet34 \
#       pgd

set -euo pipefail

FRAMES_DIR="${1:?Error: FRAMES_DIR not set. Usage: $0 <FRAMES_DIR> <WORK_DIR> <MODEL_DIR> <PERTURBATION> [CODE_DIR] [CHUNK_SIZE] [MODE_ANALYSIS]}"
WORK_DIR="${2:?Error: WORK_DIR not set.}"
MODEL_DIR="${3:?Error: MODEL_DIR not set.}"
PERTURBATION="${4:?Error: PERTURBATION not set (e.g. 'pgd').}"
CODE_DIR="${5:-$(cd "$(dirname "$0")/.." && pwd)}"
CHUNK_SIZE="${6:-20}"
MODE_ANALYSIS="${7:-1}"

CONCAT_FILE="$WORK_DIR/live_pert_concat.npz"
PARTIALS_DIR="$WORK_DIR/partials/mode_${MODE_ANALYSIS}"
PROFILES_OUT="$WORK_DIR/live_pert_profiles_${MODE_ANALYSIS}.npy"

# Upper bound on frame count — tasks past the actual data exit cleanly.
# Matches MAX_LIVE_PERT_SIZE = 200 in atoms_config.py.
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
echo "N_TASKS       : $N_TASKS (indices 0–$N_LAST)"
echo ""

mkdir -p "$WORK_DIR/logs" "$PARTIALS_DIR"

# --- Job 1: concatenate live-pert run files ---
PREP_JOB_ID=$(sbatch --parsable \
    --chdir="$CODE_DIR" \
    --export=ALL,FRAMES_DIR="$FRAMES_DIR",PERTURBATION="$PERTURBATION",CONCAT_FILE="$CONCAT_FILE",CODE_DIR="$CODE_DIR" \
    "$CODE_DIR/hpc/prep_live_pert_task.sh")
echo "Submitted prep job  : $PREP_JOB_ID"

# --- Job 2: parallel ATOMs (depends on prep) ---
ARRAY_JOB_ID=$(sbatch --parsable \
    --array=0-${N_LAST} \
    --dependency=afterok:${PREP_JOB_ID} \
    --chdir="$CODE_DIR" \
    --export=ALL,CONCAT_FILE="$CONCAT_FILE",PARTIALS_DIR="$PARTIALS_DIR",MODEL_DIR="$MODEL_DIR",CODE_DIR="$CODE_DIR",CHUNK_SIZE="$CHUNK_SIZE",MODE_ANALYSIS="$MODE_ANALYSIS" \
    "$CODE_DIR/hpc/array_live_pert_task.sh")
echo "Submitted array job : $ARRAY_JOB_ID  (${N_TASKS} tasks, indices 0–${N_LAST})"

# --- Job 3: gather (depends on all array tasks) ---
GATHER_JOB_ID=$(sbatch --parsable \
    --dependency=afterok:${ARRAY_JOB_ID} \
    --chdir="$CODE_DIR" \
    --export=ALL,PARTIALS_DIR="$PARTIALS_DIR",PROFILES_OUT="$PROFILES_OUT",CODE_DIR="$CODE_DIR",MODE_ANALYSIS="$MODE_ANALYSIS" \
    "$CODE_DIR/hpc/gather_live_pert_task.sh")
echo "Submitted gather job: $GATHER_JOB_ID"

echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f $WORK_DIR/logs/chunk_${ARRAY_JOB_ID}_0.out"
echo ""
echo "After gather completes, on Viper:"
echo "  PERT=$PERTURBATION"
echo "  ATT=/u/\$USER/pcla/data/TFV6/test_data/attention/live_pert/\$PERT"
echo "  mkdir -p \$ATT"
echo "  cp $PROFILES_OUT                                          \$ATT/live_pert_profiles_${MODE_ANALYSIS}.npy"
echo "  cp $WORK_DIR/live_pert_speed_logits_${MODE_ANALYSIS}.npy \$ATT/live_pert_speed_logits_${MODE_ANALYSIS}.npy"
echo "  cd /u/\$USER/pcla"
echo "  git add -f data/TFV6/test_data/attention/live_pert/\$PERT/live_pert_profiles_${MODE_ANALYSIS}.npy"
echo "  git add -f data/TFV6/test_data/attention/live_pert/\$PERT/live_pert_speed_logits_${MODE_ANALYSIS}.npy"
echo "  git commit -m 'add live_pert_profiles_${MODE_ANALYSIS} for \$PERT from HPC'"
echo "  git push"
echo "Then locally: git pull, set RECOMPUTE_TEST_ATOMS=False in atoms_config.py"
