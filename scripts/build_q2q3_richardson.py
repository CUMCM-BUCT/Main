"""Build the fast Richardson-validated Q2/Q3 formal CSV package."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from a_model.q2q3_richardson import build_formal_export


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=ROOT / "data/processed/q2q3/convergence_graded_v2")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--q2-end", type=float, default=10800.0)
    args = parser.parse_args()
    root = args.matrix
    report = build_formal_export(
        space_triplet=(root / "N160_dt7p5_grade2", root / "N320_dt7p5_grade2", root / "N640_dt7p5_grade2"),
        time_triplet=(root / "N320_dt15_grade2", root / "N320_dt7p5_grade2", root / "N320_dt3p75_grade2"),
        output=args.output,
        q2_end_time_s=args.q2_end,
    )
    print(report["formal_event_time_s"])


if __name__ == "__main__":
    main()
