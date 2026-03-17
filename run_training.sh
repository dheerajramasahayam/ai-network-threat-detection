#!/usr/bin/env bash
set -euo pipefail

python3 evaluation/generate_architecture.py
python3 training/run_advanced_research.py "$@"
python3 evaluation/generate_paper.py
