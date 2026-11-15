#!/usr/bin/env bash
# Start the planner against the local Bionic server on port 1235.
set -eu
cd "$(dirname "$0")"

if [[ ! -x .venv/bin/python ]]; then
    echo "Create the repo's .venv and install requirements.txt first. See Run on macOS in README.md." >&2
    exit 1
fi

base_url=http://127.0.0.1:1235/v1
export PLANNER_SINGLE_MODEL=1
export PLANNER_EMBEDDING_MODEL=qwen3-embedding-0.6b-live
export PLANNER_EMBEDDING_MODEL_KEY=text-embedding-qwen3-embedding-0.6b
export PLANNER_EMBEDDING_ENDPOINT="$base_url/embeddings"
export PLANNER_DECISION_MODEL=kev-4b-live
export PLANNER_DECISION_MODEL_KEY=kev-4b
export PLANNER_DECISION_ENDPOINT="$base_url/systemone"
export PLANNER_MODEL=ternary-bonsai-2-4b-mlx
export PLANNER_MODEL_KEY="$PLANNER_MODEL"
export PLANNER_MODEL_ENDPOINT="$base_url/chat/completions"
export PLANNER_MODEL_TIMEOUT_SECONDS="${PLANNER_MODEL_TIMEOUT_SECONDS:-360}"
export PLANNER_DATA_DIR="${PLANNER_DATA_DIR:-$HOME/Library/Application Support/Plan Smarter/lmstudio-live}"

case "${1:-}" in
    --check) exec .venv/bin/python -m examples.estimation.check_local_models ;;
    "") exec .venv/bin/python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 ;;
    *) echo "Usage: bash run-bionic-mac.sh [--check]" >&2; exit 2 ;;
esac
