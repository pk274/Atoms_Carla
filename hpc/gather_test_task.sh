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

_mode="${MODE_ANALYSIS:-1}"
# Allow the submit script to override the logits filename (e.g. val_speed_logits_N.npy for val jobs).
# Falls back to the standard test name when not set.
if [ -z "${SPEED_LOGITS_OUT:-}" ]; then
    SPEED_LOGITS_OUT="$(dirname "$PROFILES_OUT")/test_speed_logits_${_mode}.npy"
fi

srun python3 "$CODE_DIR/hpc/gather_test.py" \
    --partials-dir        "$PARTIALS_DIR" \
    --output              "$PROFILES_OUT" \
    --speed-logits-output "$SPEED_LOGITS_OUT"

echo "Gather finished with exit code $?"
echo "test_speed_logits_${_mode}.npy is at: $SPEED_LOGITS_OUT"

# Visualise perturbation samples.  Prefer the explicit LABELED_FILE env var (set by both
# submit_test.sh and submit_val.sh); fall back to the legacy name-derived path.
_LABELED="${LABELED_FILE:-$(dirname "$PROFILES_OUT")/test_labeled.npz}"
_VIZ_OUT="$(dirname "$PROFILES_OUT")/perturb_samples.png"
if [ -f "$_LABELED" ]; then
    srun python3 "$CODE_DIR/hpc/visualize_perturb.py" \
        --labeled-file "$_LABELED" \
        --output       "$_VIZ_OUT"
    echo "Perturbation samples saved: $_VIZ_OUT"
else
    echo "WARNING: $_LABELED not found — skipping perturbation visualisation."
fi

date
