#!/bin/bash -l
# array_test_task.sh
# Per-task script for the test-set ATOMs SLURM array job.
# Each task processes CHUNK_SIZE consecutive frames from test_labeled.npz.
#
# Variables injected by submit_test.sh via --export:
#   LABELED_FILE  path to test_labeled.npz
#   PARTIALS_DIR  output directory for partial_test_N.npz files
#   MODEL_DIR     path to the TFV6 pretrained model directory
#   CODE_DIR      project root
#   CHUNK_SIZE    number of frames per task

#SBATCH -J atoms_test
#SBATCH -o /ptmp/%u/atoms_test/logs/chunk_%A_%a.out
#SBATCH -e /ptmp/%u/atoms_test/logs/chunk_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000MB
#SBATCH --time=24:00:00
# Add your account/partition here if required, e.g.:
# #SBATCH --account=YOUR_ACCOUNT

module purge
module load python-waterboa/2025.06
source /u/$USER/venvs/pcla/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONPATH="$CODE_DIR/hpc/stubs:$CODE_DIR:$CODE_DIR/pcla_agents/transfuserv6:$PYTHONPATH"

mkdir -p "$PARTIALS_DIR" /ptmp/$USER/atoms_test/logs

CHUNK_START=$(( SLURM_ARRAY_TASK_ID * CHUNK_SIZE ))
CHUNK_END=$(( CHUNK_START + CHUNK_SIZE ))
OUTPUT="$PARTIALS_DIR/partial_test_${SLURM_ARRAY_TASK_ID}.npz"

echo "=== Test Task $SLURM_ARRAY_TASK_ID ==="
echo "Frames   : $CHUNK_START .. $CHUNK_END"
echo "Output   : $OUTPUT"
echo "Node     : $(hostname)"
echo "CPUs     : $SLURM_CPUS_PER_TASK"
date

# PGD attack settings for 'pgd' test frames (TFV6).
# PGD_TARGET and PGD_STEPS can be overridden by exporting before submitting.
# PGD_EPSILON has no effect here — the ε budget is read from test_labeled.npz
# (written by the prep step); export PGD_EPSILON before the *prep job* to change it.
srun python3 "$CODE_DIR/hpc/compute_test_chunk.py" \
    --labeled-file  "$LABELED_FILE"        \
    --chunk-start   "$CHUNK_START"         \
    --chunk-end     "$CHUNK_END"           \
    --output        "$OUTPUT"              \
    --model-dir     "$MODEL_DIR"           \
    --mode-analysis "${MODE_ANALYSIS:-1}"  \
    --pgd-target    "${PGD_TARGET:-steer_right}" \
    --pgd-epsilon   "${PGD_EPSILON:-12.0}"       \
    --pgd-steps     "${PGD_STEPS:-10}"

echo "Task $SLURM_ARRAY_TASK_ID finished with exit code $?"
date
