#!/usr/bin/env bash
set -euo pipefail

EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd "${EXAMPLE_DIR}/../.." && pwd)"

cd "${REPOSITORY_ROOT}"
export MPLBACKEND=Agg

python examples/synthetic_basin/build_inputs.py
python scripts/run_workflow.py \
  --config examples/synthetic_basin/config.yml \
  --stage all
python examples/synthetic_basin/check_outputs.py

echo "Synthetic example complete: examples/synthetic_basin/outputs"
