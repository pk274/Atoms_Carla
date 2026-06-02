#!/bin/bash -l
# gather_test_task.sh
# Single-node SLURM job submitted with --dependency=afterok:<array_job_id>.
# Concatenates partial profile files into test_profiles.npy.
#
# Variables injected by submit_test.sh via --export:
#   PARTIALS_DIR  directory containing partial_test_*.npz files
#   PROFILES_OUT  output path for test_profiles.npy
#   CODE_DIR      project root

#SBATCH -J atoms_gather_test
#SBATCH -o /ptmp/%u/atoms_test/logs/gather_test_%j.out
#SBATCH -e /ptmp/%u/atoms_test/logs/gather_test_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=80000MB
#SBATCH --time=05:00:00
# Add your account/partition here if required, e.g.:
# #SBATCH --account=YOUR_ACCOUNT

module purge
module load python-waterboa/2025.06
source /u/$USER/venvs/pcla/bin/activate

export PYTHONPATH="$CODE_DIR:$PYTHONPATH"

echo "=== Gather test job ==="
echo "Partials dir : $PARTIALS_DIR"
echo "Output       : $PROFILES_OUT"
echo "Node         : $(hostname)"
date

SPEED_LOGITS_OUT="$(dirname "$PROFILES_OUT")/test_speed_logits.npy"

srun python3 "$CODE_DIR/hpc/gather_test.py" \
    --partials-dir        "$PARTIALS_DIR" \
    --output              "$PROFILES_OUT" \
    --speed-logits-output "$SPEED_LOGITS_OUT"

echo "Gather finished with exit code $?"
echo "test_speed_logits.npy is at: $SPEED_LOGITS_OUT"
date
