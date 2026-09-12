"""Run the complete Q1 phase-4 convergence matrix and formal export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from a_model.inputs import load_environment  # noqa: E402
from a_model.q1_phase4 import ConvergenceGateError, execute_phase4  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "q1" / "phase4",
    )
    parser.add_argument(
        "--expected-commit",
        help="Optional full or abbreviated HEAD expected for this formal run.",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help=(
            "Allow candidate calculation from dirty source; formal outputs remain "
            "blocked."
        ),
    )
    args = parser.parse_args()
    try:
        result = execute_phase4(
            args.output,
            load_environment(),
            repo_root=REPO_ROOT,
            expected_commit=args.expected_commit,
            allow_dirty=args.allow_dirty,
        )
    except ConvergenceGateError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
