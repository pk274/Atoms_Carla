#!/bin/bash -l
# prep_test_task.sh
# Single-node SLURM job: apply perturbations to clean test frames.
# Fast (image-space ops only, no model loading).
#
# Variables injected by submit_test.sh via --export:
#   FRAMES_DIR    directory containing clean run_*.npz test frame files
#   LABELED_FILE  output path for test_labeled.npz
#   CODE_DIR      project root

#SBATCH -J atoms_prep_test
#SBATCH -o /ptmp/%u/atoms_test/logs/prep_%j.out
#SBATCH -e /ptmp/%u/atoms_test/logs/prep_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000MB
#SBATCH --time=01:00:00
# Add your account/partition here if required, e.g.:
# #SBATCH --account=YOUR_ACCOUNT

module purge
module load python-waterboa/2025.06
source /u/$USER/venvs/pcla/bin/activate

export PYTHONPATH="$CODE_DIR/hpc/stubs:$CODE_DIR:$PYTHONPATH"

echo "=== Prep test job ==="
echo "Frames dir   : $FRAMES_DIR"
echo "Output file  : $LABELED_FILE"
echo "Node         : $(hostname)"
date

srun python3 "$CODE_DIR/hpc/prep_test.py" \
    --frames-dir "$FRAMES_DIR" \
    --output     "$LABELED_FILE" \
    --seed       42

echo "Prep finished with exit code $?"
date
