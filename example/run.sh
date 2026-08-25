#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPOSITORY_ROOT}"
export MPLBACKEND=Agg

python scripts/run_workflow.py --config example/config.yml --stage all
python example/check_results.py --skip-hashes example/run

echo "Green Valley example complete: example/run"
