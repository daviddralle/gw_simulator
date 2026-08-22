#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gw_simulator.workflow import load_workflow_config, preflight_workflow, write_preflight


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate workflow inputs and estimate groundwater grid cost."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", default=None, type=Path)
    args = parser.parse_args()

    config = load_workflow_config(args.config)
    report = preflight_workflow(config)
    output = args.output or config.path_value("output_dir") / "preflight.json"
    write_preflight(report, output)
    print(json.dumps(report, indent=2))
    if not report["ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
