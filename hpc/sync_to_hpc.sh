#!/bin/bash
# sync_to_hpc.sh
# --------------
# Sync code, model weights, and baseline frames to MPCDF Viper.
# Run this from the project root on your local machine.
#
# Usage:
#   bash hpc/sync_to_hpc.sh <HPC_USERNAME>
#
# Example:
#   bash hpc/sync_to_hpc.sh pkull

set -euo pipefail

HPC_USER="${1:?Error: HPC username required. Usage: $0 <HPC_USERNAME>}"
HPC_HOST="viper.mpcdf.mpg.de"
REMOTE_CODE="/u/$HPC_USER/pcla"
REMOTE_FRAMES="/ptmp/$HPC_USER/atoms_baseline/frames"

echo "=== Syncing to $HPC_USER@$HPC_HOST ==="
echo ""

# 1. Code (exclude large data dirs, caches, PDFs)
echo "[1/3] Syncing code → $REMOTE_CODE ..."
rsync -avz --progress \
    --exclude='data/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.git/' \
    --exclude='*.pdf' \
    --exclude='*.egg-info/' \
    --exclude='.venv/' \
    --exclude='venv/' \
    --exclude='*.out' \
    --exclude='*.err' \
    ./ "$HPC_USER@$HPC_HOST:$REMOTE_CODE/"

# 2. Model weights (only TFV6; add WoR path if needed)
echo ""
echo "[2/3] Syncing TFV6 model weights → $REMOTE_CODE/pcla_agents/transfuserv6_pretrained/ ..."
rsync -avz --progress \
    pcla_agents/transfuserv6_pretrained/ \
    "$HPC_USER@$HPC_HOST:$REMOTE_CODE/pcla_agents/transfuserv6_pretrained/"

# 3. Baseline frames
echo ""
echo "[3/3] Syncing baseline frames → $REMOTE_FRAMES ..."
ssh "$HPC_USER@$HPC_HOST" "mkdir -p $REMOTE_FRAMES"
rsync -avz --progress \
    data/TFV6/baseline_data/frames/ \
    "$HPC_USER@$HPC_HOST:$REMOTE_FRAMES/"

echo ""
echo "=== Sync complete ==="
echo ""
echo "Next steps on the HPC (login with: ssh $HPC_USER@$HPC_HOST):"
echo ""
echo "  # One-time: create the Python venv"
echo "  bash $REMOTE_CODE/hpc/setup_venv.sh"
echo ""
echo "  # Submit the baseline computation"
echo "  bash $REMOTE_CODE/hpc/submit_baseline.sh \\"
echo "      $REMOTE_FRAMES \\"
echo "      /ptmp/$HPC_USER/atoms_baseline/partials \\"
echo "      $REMOTE_CODE/pcla_agents/transfuserv6_pretrained/visiononly_resnet34 \\"
echo "      $REMOTE_CODE"
echo ""
echo "  # After gather job completes, copy baseline.npz back:"
echo "  # (run this on your local machine)"
echo "  rsync -avz $HPC_USER@$HPC_HOST:/ptmp/$HPC_USER/atoms_baseline/partials/baseline.npz data/TFV6/baseline_data/"
