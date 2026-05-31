#!/bin/bash
# submit_test.sh
# --------------
# Submit the full test-set ATOMs pipeline as three chained SLURM jobs:
#
#   Job 1 (prep)   : apply perturbations to clean frames → test_labeled.npz
#   Job 2 (array)  : parallel LRP + ATOMs, one task per CHUNK_SIZE frames
#   Job 3 (gather) : concatenate partial profiles → test_profiles.npy
#
# Usage (from $CODE_DIR on the HPC):
#   bash hpc/submit_test.sh <FRAMES_DIR> <WORK_DIR> <MODEL_DIR> [CODE_DIR] [CHUNK_SIZE]
#
# Arguments:
#   FRAMES_DIR   directory containing clean run_*.npz test frame files
#                e.g. /ptmp/$USER/atoms_test/frames
#   WORK_DIR     working directory for all outputs (labeled file, partials, logs)
#                e.g. /ptmp/$USER/atoms_test
#   MODEL_DIR    path to TFV6 pretrained model directory
#                e.g. /u/$USER/pcla/pcla_agents/transfuserv6_pretrained/visiononly_resnet34
#   CODE_DIR     project root (default: parent of this script)
#   CHUNK_SIZE   frames per array task (default: 20; 10 tasks for 200 frames)
#
# Example:
#   bash hpc/submit_test.sh \
#       /ptmp/$USER/atoms_test/frames \
#       /ptmp/$USER/atoms_test \
#       /u/$USER/pcla/pcla_agents/transfuserv6_pretrained/visiononly_resnet34

set -euo pipefail

FRAMES_DIR="${1:?Error: FRAMES_DIR not set. Usage: $0 <FRAMES_DIR> <WORK_DIR> <MODEL_DIR> [CODE_DIR] [CHUNK_SIZE]}"
WORK_DIR="${2:?Error: WORK_DIR not set.}"
MODEL_DIR="${3:?Error: MODEL_DIR not set.}"
CODE_DIR="${4:-$(cd "$(dirname "$0")/.." && pwd)}"
CHUNK_SIZE="${5:-20}"

LABELED_FILE="$WORK_DIR/test_labeled.npz"
PARTIALS_DIR="$WORK_DIR/partials"
PROFILES_OUT="$WORK_DIR/test_profiles.npy"

# Upper bound on frame count — tasks past the actual data exit cleanly
MAX_FRAMES=200
N_TASKS=$(( (MAX_FRAMES + CHUNK_SIZE - 1) / CHUNK_SIZE ))
N_LAST=$(( N_TASKS - 1 ))

echo "=== ATOMs Test SLURM Submission ==="
echo "FRAMES_DIR   : $FRAMES_DIR"
echo "WORK_DIR     : $WORK_DIR"
echo "MODEL_DIR    : $MODEL_DIR"
echo "CODE_DIR     : $CODE_DIR"
echo "CHUNK_SIZE   : $CHUNK_SIZE"
echo "N_TASKS      : $N_TASKS (indices 0–$N_LAST)"
echo ""

mkdir -p "$WORK_DIR/logs" "$PARTIALS_DIR"

# --- Job 1: apply perturbations ---
PREP_JOB_ID=$(sbatch --parsable \
    --chdir="$CODE_DIR" \
    --export=ALL,FRAMES_DIR="$FRAMES_DIR",LABELED_FILE="$LABELED_FILE",CODE_DIR="$CODE_DIR" \
    "$CODE_DIR/hpc/prep_test_task.sh")
echo "Submitted prep job  : $PREP_JOB_ID"

# --- Job 2: parallel ATOMs (depends on prep) ---
ARRAY_JOB_ID=$(sbatch --parsable \
    --array=0-${N_LAST} \
    --dependency=afterok:${PREP_JOB_ID} \
    --chdir="$CODE_DIR" \
    --export=ALL,LABELED_FILE="$LABELED_FILE",PARTIALS_DIR="$PARTIALS_DIR",MODEL_DIR="$MODEL_DIR",CODE_DIR="$CODE_DIR",CHUNK_SIZE="$CHUNK_SIZE" \
    "$CODE_DIR/hpc/array_test_task.sh")
echo "Submitted array job : $ARRAY_JOB_ID  (${N_TASKS} tasks, indices 0–${N_LAST})"

# --- Job 3: gather (depends on all array tasks) ---
GATHER_JOB_ID=$(sbatch --parsable \
    --dependency=afterok:${ARRAY_JOB_ID} \
    --chdir="$CODE_DIR" \
    --export=ALL,PARTIALS_DIR="$PARTIALS_DIR",PROFILES_OUT="$PROFILES_OUT",CODE_DIR="$CODE_DIR" \
    "$CODE_DIR/hpc/gather_test_task.sh")
echo "Submitted gather job: $GATHER_JOB_ID"

echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f $WORK_DIR/logs/chunk_${ARRAY_JOB_ID}_0.out"
echo ""
echo "After gather completes, get test_profiles.npy back locally:"
echo "  cp $PROFILES_OUT /u/\$USER/pcla/data/TFV6/test_data/attention/test_profiles.npy"
echo "  cd /u/\$USER/pcla"
echo "  git add -f data/TFV6/test_data/attention/test_profiles.npy"
echo "  git commit -m 'add TFV6 test_profiles.npy from HPC'"
echo "  git push"
