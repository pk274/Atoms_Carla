#!/bin/bash -l
# array_live_pert_task.sh
# Per-task script for the live-perturbation ATOMs SLURM array job.
# Each task processes CHUNK_SIZE consecutive frames from live_pert_concat.npz.
#
# Variables injected by submit_live_pert.sh via --export:
#   CONCAT_FILE   path to live_pert_concat.npz
#   PARTIALS_DIR  output directory for partial_live_pert_N.npz files
#   MODEL_DIR     path to the TFV6 pretrained model directory
#   CODE_DIR      project root
#   CHUNK_SIZE    number of frames per task

#SBATCH -J atoms_live_pert
#SBATCH -o /ptmp/%u/atoms_live_pert/logs/chunk_%A_%a.out
#SBATCH -e /ptmp/%u/atoms_live_pert/logs/chunk_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000MB
#SBATCH --time=02:00:00
# Add your account/partition here if required, e.g.:
# #SBATCH --account=YOUR_ACCOUNT

module purge
module load python-waterboa/2025.06
source /u/$USER/venvs/pcla/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONPATH="$CODE_DIR/hpc/stubs:$CODE_DIR:$PYTHONPATH"

mkdir -p "$PARTIALS_DIR" /ptmp/$USER/atoms_live_pert/logs

CHUNK_START=$(( SLURM_ARRAY_TASK_ID * CHUNK_SIZE ))
CHUNK_END=$(( CHUNK_START + CHUNK_SIZE ))
OUTPUT="$PARTIALS_DIR/partial_live_pert_${SLURM_ARRAY_TASK_ID}.npz"

echo "=== Live-pert Task $SLURM_ARRAY_TASK_ID ==="
echo "Frames   : $CHUNK_START .. $CHUNK_END"
echo "Output   : $OUTPUT"
echo "Node     : $(hostname)"
echo "CPUs     : $SLURM_CPUS_PER_TASK"
date

srun python3 "$CODE_DIR/hpc/compute_live_pert_chunk.py" \
    --concat-file  "$CONCAT_FILE"  \
    --chunk-start  "$CHUNK_START"  \
    --chunk-end    "$CHUNK_END"    \
    --output       "$OUTPUT"       \
    --model-dir    "$MODEL_DIR"

echo "Task $SLURM_ARRAY_TASK_ID finished with exit code $?"
date
