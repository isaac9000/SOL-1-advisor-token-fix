#!/usr/bin/env bash
set -e

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

if [[ -f .env ]]; then
    set -a; source .env; set +a
fi

echo "=== attn_bwd advisor-worker optimization ==="
echo "Deploying evaluator (no-op if already deployed)..."
uv run modal deploy eval_modal_attn_bwd.py

echo ""
echo "Launching agent..."
uv run attn_bwd/agent.py \
    --baseline attn_bwd/starting_point.py \
    --iterations 25 \
    "$@"
