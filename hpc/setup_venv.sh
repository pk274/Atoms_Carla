#!/bin/bash
# setup_venv.sh
# -------------
# One-time setup: create a Python virtual environment on Viper with all
# packages needed for the HPC baseline computation.
# Run this interactively on the login node (not as a SLURM job).
#
# Usage:
#   bash hpc/setup_venv.sh

set -euo pipefail

VENV_DIR="/u/$USER/venvs/pcla"

echo "=== Setting up Python venv at $VENV_DIR ==="

module purge
module load python-waterboa/2025.06

if [ -d "$VENV_DIR" ]; then
    echo "Venv already exists at $VENV_DIR. Activating and upgrading packages..."
else
    python3 -m venv "$VENV_DIR"
    echo "Venv created."
fi

source "$VENV_DIR/bin/activate"
pip install --upgrade pip

echo ""
echo "Installing packages from hpc/requirements_hpc.txt..."
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
pip install -r "$SCRIPT_DIR/requirements_hpc.txt"

echo ""
echo "=== Setup complete ==="
echo "Python: $(python3 --version)"
echo "Torch:  $(python3 -c 'import torch; print(torch.__version__)')"
echo "Zennit: $(python3 -c 'import zennit; print(zennit.__version__)')"
echo ""
echo "To activate in future sessions:"
echo "  module load python-waterboa/2025.06 && source $VENV_DIR/bin/activate"
