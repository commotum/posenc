from __future__ import annotations

import json

from core.cli import parse_args
from encoders import encoder_names
from experiment import enforce_requested_requirements, requirement_report, run_experiment


def main() -> None:
    cfg = parse_args()
    checks = requirement_report(cfg)

    print("Encoder compatibility checks:")
    for name in encoder_names():
        check = checks[name]
        print(f"  - {name:<9} {'OK' if check.ok else 'FAIL'} | {check.rule}")

    enforce_requested_requirements(cfg.encoders, checks)
    artifacts = run_experiment(cfg)
    print(json.dumps(artifacts.summary, indent=2))


if __name__ == "__main__":
    main()
