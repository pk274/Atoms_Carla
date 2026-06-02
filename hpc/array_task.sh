#!/bin/bash -l
# array_task.sh
# Per-task script called by the SLURM array job.
# Each task processes one run_*.npz file.
#
# Variables injected by submit_baseline.sh via --export:
#   LIST_FILE    path to the newline-separated list of run .npz files
#   PARTIALS_DIR output directory for partial_N.npz files
#   MODEL_DIR    path to the TFV6 pretrained model directory
#   CODE_DIR     project root (used for PYTHONPATH)

#SBATCH -J atoms_baseline
#SBATCH -o /ptmp/%u/atoms_baseline/logs/chunk_%A_%a.out
#SBATCH -e /ptmp/%u/atoms_baseline/logs/chunk_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000MB
#SBATCH --time=03:00:00
# Add your account/partition here if required by your allocation, e.g.:
# #SBATCH --account=YOUR_ACCOUNT

module purge
module load python-waterboa/2025.06
source /u/$USER/venvs/pcla/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# hpc/stubs must come first so its carla.py stub shadows any real carla install
export PYTHONPATH="$CODE_DIR/hpc/stubs:$CODE_DIR:$CODE_DIR/pcla_agents/transfuserv6:$PYTHONPATH"

mkdir -p "$PARTIALS_DIR" /ptmp/$USER/atoms_baseline/logs

# Resolve this task's run file (LIST_FILE is 1-indexed via sed)
RUN_FILE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$LIST_FILE")
OUTPUT="$PARTIALS_DIR/partial_${SLURM_ARRAY_TASK_ID}.npz"

echo "=== Task $SLURM_ARRAY_TASK_ID ==="
echo "Run file : $RUN_FILE"
echo "Output   : $OUTPUT"
echo "Node     : $(hostname)"
echo "CPUs     : $SLURM_CPUS_PER_TASK"
date

srun python3 "$CODE_DIR/hpc/compute_baseline_chunk.py" \
    --run-file  "$RUN_FILE"   \
    --output    "$OUTPUT"     \
    --model-dir "$MODEL_DIR"  \
    --agent     TFV6

echo "Task $SLURM_ARRAY_TASK_ID finished with exit code $?"
date
