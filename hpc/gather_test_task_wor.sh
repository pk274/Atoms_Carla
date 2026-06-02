#!/bin/bash -l
# gather_test_task_wor.sh
# Single-node SLURM job: concatenate WoR partial profile files.
# Outputs test_profiles.npy and test_logits.npy (28-dim PEOC action logits).
#
# Variables injected by submit_test_wor.sh via --export:
#   PARTIALS_DIR  directory containing partial_test_*.npz files
#   PROFILES_OUT  output path for test_profiles.npy
#   CODE_DIR      project root

#SBATCH -J atoms_wor_gather_test
#SBATCH -o /ptmp/%u/atoms_wor_test/logs/gather_%j.out
#SBATCH -e /ptmp/%u/atoms_wor_test/logs/gather_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16000MB
#SBATCH --time=01:00:00
# Add your account/partition here if required, e.g.:
# #SBATCH --account=YOUR_ACCOUNT

module purge
module load python-waterboa/2025.06
source /u/$USER/venvs/pcla/bin/activate

export PYTHONPATH="$CODE_DIR:$PYTHONPATH"

echo "=== WoR Gather test job ==="
echo "Partials dir : $PARTIALS_DIR"
echo "Output       : $PROFILES_OUT"
echo "Node         : $(hostname)"
date

LOGITS_OUT="$(dirname "$PROFILES_OUT")/test_logits.npy"

srun python3 "$CODE_DIR/hpc/gather_test.py" \
    --partials-dir        "$PARTIALS_DIR" \
    --output              "$PROFILES_OUT" \
    --speed-logits-output "$LOGITS_OUT"   \
    --agent               WOR

echo "Gather finished with exit code $?"
echo "test_logits.npy is at: $LOGITS_OUT"
date
