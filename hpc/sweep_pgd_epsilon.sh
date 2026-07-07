#!/bin/bash -l
# sweep_pgd_epsilon.sh
# Single-node SLURM job: sweep the PGD ε budget to find the smallest value at
# which >= SUCCESS_TARGET of brake-target attacks force the deployed TFV6
# controller to brake (p_brake > BRAKE_THRESHOLD).  Runs sweep_pgd_epsilon.py.
#
# This is CPU-only, matching the rest of the Viper-CPU ATOMs pipeline (the PGD
# attack is a per-frame forward/backward; torch parallelises over the allocated
# cores via OMP).  It is a one-off calibration, so it runs as ONE task rather
# than a chunked array — bump --time / --cpus-per-task if your grid is large.
#
# Usage:
#   sbatch hpc/sweep_pgd_epsilon.sh
# Override defaults by exporting before submitting, e.g.:
#   CODE_DIR=/u/$USER/pcla N_FRAMES=200 EPSILONS="1 2 4 6 8 12 16 24" \
#     sbatch hpc/sweep_pgd_epsilon.sh

#SBATCH -J pgd_eps_sweep
#SBATCH -o /ptmp/%u/atoms_test/logs/pgd_eps_sweep_%j.out
#SBATCH -e /ptmp/%u/atoms_test/logs/pgd_eps_sweep_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=32000MB
#SBATCH --time=08:00:00
# Add your account/partition here if required, e.g.:
# #SBATCH --account=YOUR_ACCOUNT

module purge
module load python-waterboa/2025.06
source /u/$USER/venvs/pcla/bin/activate

# --- Config (override via environment before sbatch) ---
CODE_DIR="${CODE_DIR:-/u/$USER/pcla}"
MODEL_DIR="${MODEL_DIR:-$CODE_DIR/pcla_agents/transfuserv6_pretrained/visiononly_resnet34}"
FRAMES_DIR="${FRAMES_DIR:-$CODE_DIR/data/TFV6/test_data_alt/frames}"
OUT="${OUT:-$CODE_DIR/data/TFV6/results_alt/pgd_epsilon_sweep}"
N_FRAMES="${N_FRAMES:-200}"
N_STEPS="${N_STEPS:-5}"
EPSILONS="${EPSILONS:-1 2 4 6 8 12 16 24}"
BRAKE_THRESHOLD="${BRAKE_THRESHOLD:-0.9}"
SUCCESS_TARGET="${SUCCESS_TARGET:-0.99}"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# sweep_pgd_epsilon.py sets sys.path itself, but exporting PYTHONPATH too keeps
# it consistent with the other HPC jobs (stubs first → carla/beartype shadowed).
export PYTHONPATH="$CODE_DIR/hpc/stubs:$CODE_DIR:$CODE_DIR/pcla_agents/transfuserv6:$PYTHONPATH"

mkdir -p /ptmp/$USER/atoms_test/logs "$(dirname "$OUT")"

echo "=== PGD epsilon sweep ==="
echo "Node        : $(hostname)"
echo "CPUs        : $SLURM_CPUS_PER_TASK"
echo "Frames dir  : $FRAMES_DIR"
echo "Model dir   : $MODEL_DIR"
echo "N frames    : $N_FRAMES   N steps: $N_STEPS"
echo "Epsilons    : $EPSILONS"
echo "Criterion   : p_brake > $BRAKE_THRESHOLD, target >= $SUCCESS_TARGET"
echo "Out stem    : $OUT"
date

srun python3 "$CODE_DIR/sweep_pgd_epsilon.py" \
    --model-dir       "$MODEL_DIR"          \
    --frames-dir      "$FRAMES_DIR"         \
    --n-frames        "$N_FRAMES"           \
    --n-steps         "$N_STEPS"            \
    --epsilons        $EPSILONS             \
    --brake-threshold "$BRAKE_THRESHOLD"    \
    --success-target  "$SUCCESS_TARGET"     \
    --device          cpu                   \
    --out             "$OUT"

echo "Sweep finished with exit code $?"
date
