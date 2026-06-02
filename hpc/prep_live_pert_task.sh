#!/bin/bash -l
# prep_live_pert_task.sh
# Single-node SLURM job: concatenate live-perturbation frame files.
# Fast (pure numpy, no model loading).
#
# Variables injected by submit_live_pert.sh via --export:
#   FRAMES_DIR      directory containing run_{PERT}_live_pert_*.npz files
#   PERTURBATION    perturbation name, e.g. "pgd"
#   CONCAT_FILE     output path for live_pert_concat.npz
#   CODE_DIR        project root

#SBATCH -J atoms_prep_live_pert
#SBATCH -o /ptmp/%u/atoms_live_pert/logs/prep_%j.out
#SBATCH -e /ptmp/%u/atoms_live_pert/logs/prep_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16000MB
#SBATCH --time=01:30:00
# Add your account/partition here if required, e.g.:
# #SBATCH --account=YOUR_ACCOUNT

module purge
module load python-waterboa/2025.06
source /u/$USER/venvs/pcla/bin/activate

export PYTHONPATH="$CODE_DIR:$PYTHONPATH"

echo "=== Prep live-pert job ==="
echo "Frames dir   : $FRAMES_DIR"
echo "Perturbation : $PERTURBATION"
echo "Output file  : $CONCAT_FILE"
echo "Node         : $(hostname)"
date

srun python3 "$CODE_DIR/hpc/prep_live_pert.py" \
    --frames-dir   "$FRAMES_DIR"   \
    --perturbation "$PERTURBATION" \
    --output       "$CONCAT_FILE"

echo "Prep finished with exit code $?"
date
