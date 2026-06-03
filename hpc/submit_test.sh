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
#   CHUNK_SIZE   frames per array task (default: 20; N_TASKS derived from actual frame count)
#
# Example:
#   bash hpc/submit_test.sh \
#       /ptmp/$USER/atoms_test/frames \
#       /ptmp/$USER/atoms_test \
#       /u/$USER/pcla/pcla_agents/transfuserv6_pretrained/visiononly_resnet34

set -euo pipefail

FRAMES_DIR="${1:?Error: FRAMES_DIR not set. Usage: $0 <FRAMES_DIR> <WORK_DIR> <MODEL_DIR> [CODE_DIR] [CHUNK_SIZE] [MODE_ANALYSIS]}"
WORK_DIR="${2:?Error: WORK_DIR not set.}"
MODEL_DIR="${3:?Error: MODEL_DIR not set.}"
CODE_DIR="${4:-$(cd "$(dirname "$0")/.." && pwd)}"
CHUNK_SIZE="${5:-20}"
MODE_ANALYSIS="${6:-1}"

LABELED_FILE="$WORK_DIR/test_labeled.npz"
PARTIALS_DIR="$WORK_DIR/partials/mode_${MODE_ANALYSIS}"
PROFILES_OUT="$WORK_DIR/test_profiles_${MODE_ANALYSIS}.npy"

# Count frames dynamically from the clean run_*.npz files in FRAMES_DIR
N_FRAMES=$(python3 -c "
import numpy as np, pathlib, sys
files = sorted(pathlib.Path('${FRAMES_DIR}').glob('run_*.npz'))
if not files:
    sys.exit('Error: no run_*.npz files found in ${FRAMES_DIR}')
total = sum(np.load(str(f), allow_pickle=False)['frame_idx'].shape[0] for f in files)
print(total)
")
N_TASKS=$(( (N_FRAMES + CHUNK_SIZE - 1) / CHUNK_SIZE ))
N_LAST=$(( N_TASKS - 1 ))

echo "=== ATOMs Test SLURM Submission ==="
echo "FRAMES_DIR    : $FRAMES_DIR"
echo "WORK_DIR      : $WORK_DIR"
echo "MODEL_DIR     : $MODEL_DIR"
echo "CODE_DIR      : $CODE_DIR"
echo "CHUNK_SIZE    : $CHUNK_SIZE"
echo "MODE_ANALYSIS : $MODE_ANALYSIS"
echo "N_FRAMES      : $N_FRAMES"
echo "N_TASKS       : $N_TASKS (indices 0–$N_LAST)"
echo ""

mkdir -p "$WORK_DIR/logs" "$PARTIALS_DIR"

# --- Job 1: apply perturbations ---
# Skip if test_labeled.npz already exists — it is mode-independent, so a second
# mode submission can reuse the file from the first run without risk of corruption.
if [ -f "$LABELED_FILE" ]; then
    echo "test_labeled.npz already exists — skipping prep job."
    ARRAY_DEP=""
else
    PREP_JOB_ID=$(sbatch --parsable \
        --chdir="$CODE_DIR" \
        --export=ALL,FRAMES_DIR="$FRAMES_DIR",LABELED_FILE="$LABELED_FILE",CODE_DIR="$CODE_DIR" \
        "$CODE_DIR/hpc/prep_test_task.sh")
    echo "Submitted prep job  : $PREP_JOB_ID"
    ARRAY_DEP="--dependency=afterok:${PREP_JOB_ID}"
fi

# --- Job 2: parallel ATOMs (depends on prep if it was submitted) ---
ARRAY_JOB_ID=$(sbatch --parsable \
    --array=0-${N_LAST} \
    ${ARRAY_DEP} \
    --chdir="$CODE_DIR" \
    --export=ALL,LABELED_FILE="$LABELED_FILE",PARTIALS_DIR="$PARTIALS_DIR",MODEL_DIR="$MODEL_DIR",CODE_DIR="$CODE_DIR",CHUNK_SIZE="$CHUNK_SIZE",MODE_ANALYSIS="$MODE_ANALYSIS" \
    "$CODE_DIR/hpc/array_test_task.sh")
echo "Submitted array job : $ARRAY_JOB_ID  (${N_TASKS} tasks, indices 0–${N_LAST})"

# --- Job 3: gather (depends on all array tasks) ---
GATHER_JOB_ID=$(sbatch --parsable \
    --dependency=afterok:${ARRAY_JOB_ID} \
    --chdir="$CODE_DIR" \
    --export=ALL,PARTIALS_DIR="$PARTIALS_DIR",PROFILES_OUT="$PROFILES_OUT",CODE_DIR="$CODE_DIR",MODE_ANALYSIS="$MODE_ANALYSIS" \
    "$CODE_DIR/hpc/gather_test_task.sh")
echo "Submitted gather job: $GATHER_JOB_ID"

echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f $WORK_DIR/logs/chunk_${ARRAY_JOB_ID}_0.out"
echo ""
echo "After gather completes, on Viper:"
echo "  ATT=/u/\$USER/pcla/data/TFV6/test_data/attention"
echo "  cp $PROFILES_OUT                                       \$ATT/test_profiles_${MODE_ANALYSIS}.npy"
echo "  cp $WORK_DIR/test_speed_logits_${MODE_ANALYSIS}.npy   \$ATT/test_speed_logits_${MODE_ANALYSIS}.npy"
echo "  cd /u/\$USER/pcla"
echo "  git add -f data/TFV6/test_data/attention/test_profiles_${MODE_ANALYSIS}.npy"
echo "  git add -f data/TFV6/test_data/attention/test_speed_logits_${MODE_ANALYSIS}.npy"
echo "  git commit -m 'add TFV6 test_profiles_${MODE_ANALYSIS}.npy and test_speed_logits_${MODE_ANALYSIS}.npy from HPC'"
echo "  git push"
echo "Then locally: git pull, set RECOMPUTE_TEST_ATOMS=False"
