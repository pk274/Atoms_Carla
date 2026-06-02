#!/bin/bash
# submit_baseline_wor.sh
# ----------------------
# Discover all WoR run_*.npz files, submit a SLURM array job (one task per
# file), then chain a gather job that runs after all tasks succeed.
#
# Usage (from $CODE_DIR on the HPC):
#   bash hpc/submit_baseline_wor.sh <FRAMES_DIR> <PARTIALS_DIR> <MODEL_DIR> [CODE_DIR]
#
# Arguments:
#   FRAMES_DIR   directory containing WoR run_*.npz files
#                e.g. /ptmp/$USER/atoms_wor_baseline/frames
#   PARTIALS_DIR output directory for partial results and final baseline.npz
#                e.g. /ptmp/$USER/atoms_wor_baseline/partials
#   MODEL_DIR    path to WoR pretrained weights directory
#                e.g. /u/$USER/pcla/pcla_agents/wor_pretrained/leaderboard_weights
#   CODE_DIR     project root (default: parent of this script)
#
# Example:
#   bash hpc/submit_baseline_wor.sh \
#       /ptmp/$USER/atoms_wor_baseline/frames \
#       /ptmp/$USER/atoms_wor_baseline/partials \
#       /u/$USER/pcla/pcla_agents/wor_pretrained/leaderboard_weights

set -euo pipefail

FRAMES_DIR="${1:?Error: FRAMES_DIR not set. Usage: $0 <FRAMES_DIR> <PARTIALS_DIR> <MODEL_DIR> [CODE_DIR]}"
PARTIALS_DIR="${2:?Error: PARTIALS_DIR not set.}"
MODEL_DIR="${3:?Error: MODEL_DIR not set.}"
CODE_DIR="${4:-$(cd "$(dirname "$0")/.." && pwd)}"

echo "=== WoR ATOMs Baseline SLURM Submission ==="
echo "FRAMES_DIR   : $FRAMES_DIR"
echo "PARTIALS_DIR : $PARTIALS_DIR"
echo "MODEL_DIR    : $MODEL_DIR"
echo "CODE_DIR     : $CODE_DIR"
echo ""

# Collect all run files
mapfile -t RUN_FILES < <(ls "$FRAMES_DIR"/run_*.npz 2>/dev/null | sort)
N_FILES=${#RUN_FILES[@]}

if [ "$N_FILES" -eq 0 ]; then
    echo "ERROR: No run_*.npz files found in $FRAMES_DIR"
    exit 1
fi

echo "Found $N_FILES run files."
N_LAST=$((N_FILES - 1))

mkdir -p "$PARTIALS_DIR"
LIST_FILE="$PARTIALS_DIR/run_file_list.txt"
printf '%s\n' "${RUN_FILES[@]}" > "$LIST_FILE"
echo "File list written to: $LIST_FILE"
echo ""

# Submit array job
ARRAY_JOB_ID=$(sbatch --parsable \
    --array=0-${N_LAST} \
    --chdir="$CODE_DIR" \
    --export=ALL,LIST_FILE="$LIST_FILE",PARTIALS_DIR="$PARTIALS_DIR",MODEL_DIR="$MODEL_DIR",CODE_DIR="$CODE_DIR" \
    "$CODE_DIR/hpc/array_task_wor.sh")

echo "Submitted array job: $ARRAY_JOB_ID  (${N_FILES} tasks, indices 0–${N_LAST})"

# Submit gather job with dependency on all array tasks
GATHER_JOB_ID=$(sbatch --parsable \
    --dependency=afterok:${ARRAY_JOB_ID} \
    --chdir="$CODE_DIR" \
    --export=ALL,PARTIALS_DIR="$PARTIALS_DIR",CODE_DIR="$CODE_DIR" \
    "$CODE_DIR/hpc/gather_task_wor.sh")

echo "Submitted gather job: $GATHER_JOB_ID  (runs after all array tasks succeed)"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f /ptmp/\$USER/atoms_wor_baseline/logs/chunk_${ARRAY_JOB_ID}_0.out"
echo ""
echo "After gather completes, on Viper:"
echo "  cp $PARTIALS_DIR/baseline.npz     /u/\$USER/pcla/data/WOR/baseline_data/baseline.npz"
echo "  cp $PARTIALS_DIR/mdx_features.npz /u/\$USER/pcla/data/WOR/baseline_data/mdx_features.npz"
echo "  cd /u/\$USER/pcla"
echo "  git add -f data/WOR/baseline_data/baseline.npz"
echo "  git add -f data/WOR/baseline_data/mdx_features.npz"
echo "  git commit -m 'add WOR baseline.npz and mdx_features.npz from HPC'"
echo "  git push"
echo "Then locally: git pull, set RECOMPUTE_BASELINE=False and RECOMPUTE_MDX_BASELINE=False"
