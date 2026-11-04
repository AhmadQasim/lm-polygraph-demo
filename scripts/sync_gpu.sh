#!/usr/bin/env bash
# Sync this project to/from a GPU box.
#
#   REMOTE=user@host:/path/to/journal_club scripts/sync_gpu.sh push
#   REMOTE=user@host:/path/to/journal_club scripts/sync_gpu.sh pull
#
# push sends only source (~92KB): scripts, requirements, manual labels.
# pull brings back only results (~44KB): the cached runs in outputs/.
# Models are re-downloaded from HF on the remote; nothing large travels.
# Never syncs .venv (platform-specific) or figures/ (regenerable).
set -euo pipefail

: "${REMOTE:?set REMOTE=user@host:/path/to/journal_club}"
MODE="${1:-}"
RSYNC=(rsync -avz --human-readable)

case "$MODE" in
  push)
    # No trailing slash on "scripts": copy the DIRECTORY, not its contents,
    # so the remote layout matches the local one.
    "${RSYNC[@]}" --exclude '__pycache__' scripts "$REMOTE/"
    "${RSYNC[@]}" requirements.txt manual_labels.json "$REMOTE/"
    echo
    echo "Sent. On the remote, once only:"
    echo "  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
    echo "Then e.g.:"
    echo "  .venv/bin/python scripts/run_bioasq.py --n 1000 --max-tier 4 --deberta-batch-size 64"
    ;;
  pull)
    mkdir -p outputs
    "${RSYNC[@]}" "$REMOTE/outputs/" outputs/
    echo
    echo "Pulled. Analyse and plot locally:"
    echo "  .venv/bin/python scripts/analyze_bioasq.py outputs/<file>.json --manual manual_labels.json"
    echo "  .venv/bin/python slides/plot_bioasq.py outputs/<file>.json"
    ;;
  *)
    echo "usage: REMOTE=user@host:/path scripts/sync_gpu.sh {push|pull}" >&2
    exit 2
    ;;
esac
