#!/bin/bash -l
# gather_task.sh
# Single-node SLURM job submitted with --dependency=afterok:<array_job_id>.
# Concatenates partial series files into the final baseline.npz.
#
# Variables injected by submit_baseline.sh via --export:
#   PARTIALS_DIR  directory containing partial_*.npz files
#   CODE_DIR      project root (used for PYTHONPATH)

#SBATCH -J atoms_gather
#SBATCH -o /ptmp/%u/atoms_baseline/logs/gather_%j.out
#SBATCH -e /ptmp/%u/atoms_baseline/logs/gather_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8000MB
#SBATCH --time=00:15:00
# Add your account/partition here if required by your allocation, e.g.:
# #SBATCH --account=YOUR_ACCOUNT

module purge
module load python-waterboa/2025.06
source /u/$USER/venvs/pcla/bin/activate

export PYTHONPATH="$CODE_DIR:$PYTHONPATH"

echo "=== Gather job ==="
echo "Partials dir : $PARTIALS_DIR"
echo "Node         : $(hostname)"
date

srun python3 "$CODE_DIR/hpc/gather_baseline.py" \
    --partials-dir "$PARTIALS_DIR" \
    --output       "$PARTIALS_DIR/baseline.npz"

echo "Gather finished with exit code $?"
echo "baseline.npz is at: $PARTIALS_DIR/baseline.npz"
echo "Copy it back with:"
echo "  rsync -avz \$HPC_USER@viper.mpcdf.mpg.de:$PARTIALS_DIR/baseline.npz data/TFV6/baseline_data/"
date
