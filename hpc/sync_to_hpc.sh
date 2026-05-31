#!/bin/bash
# sync_to_hpc.sh
# --------------
# Transfer frames and code to MPCDF Viper.
# Run this from the project root on your local machine.
#
# IMPORTANT — direct rsync/scp to Viper does not work; Viper is only reachable
# via gate1.mpcdf.mpg.de.  This script uses an HTTP server + SSH reverse tunnel
# to stream data from your laptop directly into Viper's /ptmp/ with no
# intermediate storage on the gateway.
#
# Usage:
#   bash hpc/sync_to_hpc.sh <HPC_USERNAME> [baseline|test|all]
#
# Arguments:
#   HPC_USERNAME   your Viper username  (e.g. paulkull)
#   MODE           which frames to upload: baseline, test, or all (default: all)
#
# Examples:
#   bash hpc/sync_to_hpc.sh paulkull all       # first-time setup
#   bash hpc/sync_to_hpc.sh paulkull test      # upload test frames only

set -euo pipefail

HPC_USER="${1:?Error: HPC username required. Usage: $0 <HPC_USERNAME> [baseline|test|all]}"
MODE="${2:-all}"
GATEWAY="gate1.mpcdf.mpg.de"
HPC_HOST="viper.mpcdf.mpg.de"
SSH_OPTS='-o "MACs=hmac-sha2-256-etm@openssh.com"'
LOCAL_HTTP_PORT=8888
REMOTE_TUNNEL_PORT=9999

REMOTE_CODE="/u/$HPC_USER/pcla"
REMOTE_BASELINE_FRAMES="/ptmp/$HPC_USER/atoms_baseline/frames"
REMOTE_TEST_FRAMES="/ptmp/$HPC_USER/atoms_test/frames"

# ---------------------------------------------------------------------------
# Step 1: sync code via git (fast, no tunnel needed)
# ---------------------------------------------------------------------------
echo "=== Step 1: Code ==="
echo "On Viper, run:"
echo "  cd $REMOTE_CODE && git pull"
echo ""

# ---------------------------------------------------------------------------
# Helper: print tunnel transfer instructions for one local directory
# ---------------------------------------------------------------------------
print_tunnel_instructions() {
    local LOCAL_DIR="$1"
    local REMOTE_DIR="$2"
    local LABEL="$3"

    echo "=== $LABEL ==="
    echo ""
    echo "  [Terminal 1 — keep open while transfer runs]"
    echo "  cd $LOCAL_DIR"
    echo "  python -m http.server $LOCAL_HTTP_PORT"
    echo ""
    echo "  [Terminal 2 — opens a shell on Viper with the reverse tunnel]"
    echo "  ssh $SSH_OPTS -R ${REMOTE_TUNNEL_PORT}:localhost:${LOCAL_HTTP_PORT} -J ${HPC_USER}@${GATEWAY} ${HPC_USER}@${HPC_HOST}"
    echo ""
    echo "  [On Viper, in that Terminal 2 shell]"
    echo "  mkdir -p $REMOTE_DIR"
    echo "  cd $REMOTE_DIR"
    echo "  wget -r -np -nd -A '*.npz' http://localhost:${REMOTE_TUNNEL_PORT}/"
    echo ""
    echo "  When wget finishes, Ctrl-C the http.server in Terminal 1."
    echo ""
}

# ---------------------------------------------------------------------------
# Print instructions for requested mode
# ---------------------------------------------------------------------------
if [[ "$MODE" == "baseline" || "$MODE" == "all" ]]; then
    print_tunnel_instructions \
        "$(pwd)/data/TFV6/baseline_data/frames" \
        "$REMOTE_BASELINE_FRAMES" \
        "Step 2: Baseline frames → $REMOTE_BASELINE_FRAMES"
fi

if [[ "$MODE" == "test" || "$MODE" == "all" ]]; then
    print_tunnel_instructions \
        "$(pwd)/data/TFV6/test_data/frames" \
        "$REMOTE_TEST_FRAMES" \
        "Step 3: Test frames → $REMOTE_TEST_FRAMES"
fi

# ---------------------------------------------------------------------------
# Next steps reminder
# ---------------------------------------------------------------------------
echo "=== After upload: submit jobs on Viper ==="
echo ""
if [[ "$MODE" == "baseline" || "$MODE" == "all" ]]; then
    echo "  # Baseline ATOMs:"
    echo "  bash $REMOTE_CODE/hpc/submit_baseline.sh \\"
    echo "      $REMOTE_BASELINE_FRAMES \\"
    echo "      /ptmp/$HPC_USER/atoms_baseline/partials \\"
    echo "      $REMOTE_CODE/pcla_agents/transfuserv6_pretrained/visiononly_resnet34 \\"
    echo "      $REMOTE_CODE"
    echo ""
fi
if [[ "$MODE" == "test" || "$MODE" == "all" ]]; then
    echo "  # Test ATOMs:"
    echo "  bash $REMOTE_CODE/hpc/submit_test.sh \\"
    echo "      $REMOTE_TEST_FRAMES \\"
    echo "      /ptmp/$HPC_USER/atoms_test \\"
    echo "      $REMOTE_CODE/pcla_agents/transfuserv6_pretrained/visiononly_resnet34 \\"
    echo "      $REMOTE_CODE"
    echo ""
fi
