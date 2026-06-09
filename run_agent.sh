#!/usr/bin/env bash
set -e
cd /workspace/trimul-advisor

# Source env for Modal/Anthropic credentials
if [[ -f .env ]]; then
    set -a; source .env; set +a
fi

echo "Checking GPU..."
OUTPUT=$(uv run python trimul/run_eval.py trimul/submission.py -o /tmp/gpu_check.json --mode test 2>&1)
echo "$OUTPUT"

GPU_LINE=$(echo "$OUTPUT" | grep "GPU:" || true)
echo ""
echo "Detected: $GPU_LINE"

if echo "$OUTPUT" | grep -q "NVIDIA H100"; then
    echo ""
    echo "--- GPU is H100 — launching agent in tmux ---"
    echo ""
    SESSION="trimul-agent"
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        tmux kill-session -t "$SESSION"
    fi
    tmux new-session -d -s "$SESSION" -c "/workspace/trimul-advisor" \
        "bash -c 'set -a && source /workspace/trimul-advisor/.env && set +a && uv run trimul/agent.py --baseline trimul/starting_point.py --iterations 25 2>&1 | tee /tmp/agent_run.log; echo; echo \"--- agent finished, press any key to exit ---\"; read -n1'"
    tmux attach-session -t "$SESSION"
else
    echo ""
    echo "--- GPU is NOT H100 — aborting ---"
    exit 1
fi
