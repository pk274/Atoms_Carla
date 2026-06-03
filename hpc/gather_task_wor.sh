#!/bin/bash -l
# gather_task_wor.sh
# Single-node SLURM job submitted with --dependency=afterok:<array_job_id>.
# Concatenates partial series files into the final WoR baseline.npz.
#
# Variables injected by submit_baseline_wor.sh via --export:
#   PARTIALS_DIR  directory containing partial_*.npz files
#   CODE_DIR      project root (used for PYTHONPATH)

#SBATCH -J atoms_wor_gather
#SBATCH -o /ptmp/%u/atoms_wor_baseline/logs/gather_%j.out
#SBATCH -e /ptmp/%u/atoms_wor_baseline/logs/gather_%j.err
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

echo "=== WoR Gather baseline job ==="
echo "Partials dir : $PARTIALS_DIR"
echo "Node         : $(hostname)"
date

_mode="${MODE_ANALYSIS:-1}"
_base="$(dirname "$PARTIALS_DIR")"
_baseline_out="$_base/baseline_${_mode}.npz"
_mdx_out="$_base/mdx_features.npz"

srun python3 "$CODE_DIR/hpc/gather_baseline.py" \
    --partials-dir "$PARTIALS_DIR" \
    --output       "$_baseline_out" \
    --mdx-output   "$_mdx_out"

echo "Gather finished with exit code $?"
echo ""
echo "baseline_${_mode}.npz : $_baseline_out"
echo "mdx_features.npz      : $_mdx_out"
echo ""
echo "Copy both into the repo and push:"
echo "  cp $_baseline_out /u/\$USER/pcla/data/WOR/baseline_data/baseline_${_mode}.npz"
echo "  cp $_mdx_out      /u/\$USER/pcla/data/WOR/baseline_data/mdx_features.npz"
echo "  cd /u/\$USER/pcla"
echo "  git add -f data/WOR/baseline_data/baseline_${_mode}.npz"
echo "  git add -f data/WOR/baseline_data/mdx_features.npz"
echo "  git commit -m 'add WOR baseline_${_mode}.npz and mdx_features.npz from HPC'"
echo "  git push"
echo ""
echo "Then locally: git pull, set RECOMPUTE_BASELINE=False and RECOMPUTE_MDX_BASELINE=False"
date
