#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/NickOLeeney/qubo-wind-farm-layout.git"
REPO_DIR="/content/qubo-wind-farm-layout"


cd "$REPO_DIR"

# 1. Init submodule thomas-wflo-benchmark
git submodule update --init --recursive

# 2. Install dependencies without version constraints (avoid Colab conflicts)
pip install -r <(sed -E 's/[<>=!~].*$//' requirements.txt)

echo ""
echo "Setup complete. Run the solver with:"
echo "  python src/qubo_windfarm_layout/solve_qubo.py --grid-resolution 500 --runtime 10"
