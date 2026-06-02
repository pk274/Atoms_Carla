#!/bin/bash -l
# gather_live_pert_task_wor.sh
# Single-node SLURM job: concatenate WoR partial live-pert profile files.
# Outputs live_pert_profiles.npy and live_pert_action_logits.npy (28-dim).
#
# Variables injected by submit_live_pert_wor.sh via --export:
#   PARTIALS_DIR   directory containing partial_live_pert_*.npz files
#   PROFILES_OUT   output path for live_pert_profiles.npy
#   CODE_DIR       project root

#SBATCH -J atoms_wor_gather_live_pert
#SBATCH -o /ptmp/%u/atoms_wor_live_pert/logs/gather_%j.out
#SBATCH -e /ptmp/%u/atoms_wor_live_pert/logs/gather_%j.err
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

echo "=== WoR Gather live-pert job ==="
echo "Partials dir : $PARTIALS_DIR"
echo "Output       : $PROFILES_OUT"
echo "Node         : $(hostname)"
date

LOGITS_OUT="$(dirname "$PROFILES_OUT")/live_pert_action_logits.npy"

srun python3 "$CODE_DIR/hpc/gather_live_pert.py" \
    --partials-dir        "$PARTIALS_DIR" \
    --output              "$PROFILES_OUT" \
    --speed-logits-output "$LOGITS_OUT"   \
    --agent               WOR

echo "Gather finished with exit code $?"
echo "live_pert_action_logits.npy is at: $LOGITS_OUT"
date
