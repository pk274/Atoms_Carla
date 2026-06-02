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
#SBATCH --mem=80000MB
#SBATCH --time=05:00:00
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
    --output       "$PARTIALS_DIR/baseline.npz" \
    --mdx-output   "$PARTIALS_DIR/mdx_features.npz"

echo "Gather finished with exit code $?"
echo ""
echo "baseline.npz     : $PARTIALS_DIR/baseline.npz"
echo "mdx_features.npz : $PARTIALS_DIR/mdx_features.npz"
echo ""
echo "Copy both into the repo and push:"
echo "  cp $PARTIALS_DIR/baseline.npz     /u/\$USER/pcla/data/TFV6/baseline_data/baseline.npz"
echo "  cp $PARTIALS_DIR/mdx_features.npz /u/\$USER/pcla/data/TFV6/baseline_data/mdx_features.npz"
echo "  cd /u/\$USER/pcla"
echo "  git add -f data/TFV6/baseline_data/baseline.npz"
echo "  git add -f data/TFV6/baseline_data/mdx_features.npz"
echo "  git commit -m 'add TFV6 baseline.npz and mdx_features.npz from HPC'"
echo "  git push"
echo ""
echo "Then locally: git pull, set RECOMPUTE_BASELINE=False and RECOMPUTE_MDX_BASELINE=False"
date
