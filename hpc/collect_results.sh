#!/bin/bash
# collect_results.sh
# ------------------
# Copy HPC computation outputs from the /ptmp work dir into the repo data/ tree
# and `git add -f` them, so you never have to cp + git add each file by hand.
# This script does NOT commit or push — it stages the files and prints the exact
# commit command for you to run.
#
# Run this on Viper, after the gather job for a pipeline has finished.
#
# Usage:
#   bash hpc/collect_results.sh <pipeline> <agent> <mode> [pert] [options]
#
#   pipeline : baseline | test | live_pert
#   agent    : tfv6 | wor
#   mode     : 1 | 2          (MODE_ANALYSIS)
#   pert     : perturbation name — REQUIRED for live_pert (e.g. pgd)
#
# Options:
#   --work-dir DIR   override the /ptmp work dir (default per pipeline+agent)
#   --code-dir DIR   repo root (default: git root of this script, else /u/$USER/pcla)
#   --no-add         copy files but skip `git add`
#   --dry-run        print what would happen without copying or staging
#
# Examples:
#   bash hpc/collect_results.sh test tfv6 1            # TFV6 test profiles+logits, mode 1
#   bash hpc/collect_results.sh test tfv6 2
#   bash hpc/collect_results.sh baseline wor 1
#   bash hpc/collect_results.sh live_pert tfv6 1 pgd
#
# Source/destination map (filenames are preserved; <logit> depends on agent):
#   baseline   : <work>/**/baseline_<mode>.npz, mdx_features.npz
#                -> data/<AGENT>/baseline_data/
#   test       : <work>/**/test_profiles_<mode>.npy, <logit>
#                -> data/<AGENT>/test_data/attention/
#   live_pert  : <work>/**/live_pert_profiles_<mode>.npy, <logit>
#                -> data/<AGENT>/test_data/attention/live_pert/<pert>/
#   <logit>: TFV6 test=test_speed_logits  WOR test=test_logits
#            TFV6 live=live_pert_speed_logits  WOR live=live_pert_action_logits

set -uo pipefail

# --------------------------------------------------------------------------- #
# Parse arguments
# --------------------------------------------------------------------------- #
PIPELINE="${1:-}"
AGENT="${2:-}"
MODE="${3:-}"
PERT=""
WORK_DIR=""
CODE_DIR=""
DO_ADD=1
DRY_RUN=0

usage() { sed -n '2,40p' "$0"; exit "${1:-1}"; }

[ -z "$PIPELINE" ] || [ -z "$AGENT" ] || [ -z "$MODE" ] && {
    echo "ERROR: pipeline, agent and mode are required." >&2; usage 1; }
shift 3 2>/dev/null || true

# Optional 4th positional = pert (only if it does not start with '-')
if [ "${1:-}" ] && [ "${1:0:2}" != "--" ]; then PERT="$1"; shift; fi

while [ "${1:-}" ]; do
    case "$1" in
        --work-dir) WORK_DIR="$2"; shift 2 ;;
        --code-dir) CODE_DIR="$2"; shift 2 ;;
        --no-add)   DO_ADD=0; shift ;;
        --dry-run)  DRY_RUN=1; shift ;;
        -h|--help)  usage 0 ;;
        *) echo "ERROR: unknown option '$1'." >&2; usage 1 ;;
    esac
done

# --------------------------------------------------------------------------- #
# Validate + normalise
# --------------------------------------------------------------------------- #
case "$PIPELINE" in baseline|test|live_pert) ;; *)
    echo "ERROR: pipeline must be baseline|test|live_pert (got '$PIPELINE')." >&2; exit 1 ;; esac
case "$AGENT" in tfv6|TFV6) AGENT_LC=tfv6; AG=TFV6 ;; wor|WOR) AGENT_LC=wor; AG=WOR ;; *)
    echo "ERROR: agent must be tfv6|wor (got '$AGENT')." >&2; exit 1 ;; esac
case "$MODE" in 1|2) ;; *)
    echo "ERROR: mode must be 1|2 (got '$MODE')." >&2; exit 1 ;; esac
if [ "$PIPELINE" = "live_pert" ] && [ -z "$PERT" ]; then
    echo "ERROR: live_pert requires a <pert> argument (e.g. pgd)." >&2; exit 1
fi

# Default work dir: /ptmp/$USER/atoms_[wor_]<pipeline>
if [ -z "$WORK_DIR" ]; then
    WOR_PREFIX=""; [ "$AGENT_LC" = "wor" ] && WOR_PREFIX="wor_"
    WORK_DIR="/ptmp/${USER}/atoms_${WOR_PREFIX}${PIPELINE}"
fi

# Default code dir: git toplevel of this script, else /u/$USER/pcla
if [ -z "$CODE_DIR" ]; then
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
    CODE_DIR="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "/u/${USER}/pcla")"
fi

# --------------------------------------------------------------------------- #
# Build the file list + destination directory
# --------------------------------------------------------------------------- #
declare -a SRC_NAMES
case "$PIPELINE" in
    baseline)
        DEST_REL="data/${AG}/baseline_data"
        SRC_NAMES=("baseline_${MODE}.npz" "mdx_features.npz")
        ;;
    test)
        DEST_REL="data/${AG}/test_data/attention"
        if [ "$AG" = "WOR" ]; then LOGIT="test_logits_${MODE}.npy"
        else                       LOGIT="test_speed_logits_${MODE}.npy"; fi
        SRC_NAMES=("test_profiles_${MODE}.npy" "$LOGIT")
        ;;
    live_pert)
        DEST_REL="data/${AG}/test_data/attention/live_pert/${PERT}"
        if [ "$AG" = "WOR" ]; then LOGIT="live_pert_action_logits_${MODE}.npy"
        else                       LOGIT="live_pert_speed_logits_${MODE}.npy"; fi
        SRC_NAMES=("live_pert_profiles_${MODE}.npy" "$LOGIT")
        ;;
esac
DEST_DIR="${CODE_DIR}/${DEST_REL}"

echo "=== collect_results: ${PIPELINE} / ${AG} / mode ${MODE}${PERT:+ / $PERT} ==="
echo "Work dir : $WORK_DIR"
echo "Code dir : $CODE_DIR"
echo "Dest     : $DEST_REL/"
[ "$DRY_RUN" = 1 ] && echo "(dry run — no files copied or staged)"
echo ""

if [ ! -d "$WORK_DIR" ]; then
    echo "ERROR: work dir does not exist: $WORK_DIR" >&2; exit 1
fi
[ "$DRY_RUN" = 0 ] && mkdir -p "$DEST_DIR"

# --------------------------------------------------------------------------- #
# Locate, copy and stage each file
# --------------------------------------------------------------------------- #
N_OK=0; N_MISS=0
declare -a STAGED_REL
for fname in "${SRC_NAMES[@]}"; do
    # Locate the source anywhere under the work dir (handles partials/mode_* nesting)
    mapfile -t HITS < <(find "$WORK_DIR" -type f -name "$fname" 2>/dev/null | sort)
    if [ "${#HITS[@]}" -eq 0 ]; then
        echo "  MISSING  $fname  (not found under $WORK_DIR)"
        N_MISS=$((N_MISS + 1)); continue
    fi
    SRC="${HITS[0]}"
    if [ "${#HITS[@]}" -gt 1 ]; then
        echo "  WARN     $fname  matched ${#HITS[@]} files; using: $SRC"
    fi
    DEST="${DEST_DIR}/${fname}"
    if [ "$DRY_RUN" = 1 ]; then
        echo "  would cp $SRC"
        echo "        -> ${DEST_REL}/${fname}"
    else
        cp -f "$SRC" "$DEST"
        echo "  copied   ${DEST_REL}/${fname}"
        if [ "$DO_ADD" = 1 ]; then
            git -C "$CODE_DIR" add -f "${DEST_REL}/${fname}"
        fi
    fi
    STAGED_REL+=("${DEST_REL}/${fname}")
    N_OK=$((N_OK + 1))
done

# --------------------------------------------------------------------------- #
# Summary + next steps
# --------------------------------------------------------------------------- #
echo ""
echo "Collected $N_OK file(s)${N_MISS:+, $N_MISS missing}."
if [ "$N_OK" -gt 0 ] && [ "$DRY_RUN" = 0 ]; then
    LABEL="${AG} ${PIPELINE}${PERT:+ $PERT} mode ${MODE}"
    if [ "$DO_ADD" = 1 ]; then
        echo ""
        echo "Staged for commit. To commit & push:"
        echo "  cd $CODE_DIR"
        echo "  git commit -m 'add ${LABEL} results from HPC'"
        echo "  git push"
    else
        echo "(--no-add: files copied but not staged)"
    fi
    case "$PIPELINE" in
        baseline)  echo "Reminder: locally set RECOMPUTE_BASELINE=False and RECOMPUTE_MDX_BASELINE=False." ;;
        test|live_pert) echo "Reminder: locally set RECOMPUTE_TEST_ATOMS=False (and REAPPLY_PERTURBATIONS=False)." ;;
    esac
fi

[ "$N_MISS" -gt 0 ] && exit 2 || exit 0
