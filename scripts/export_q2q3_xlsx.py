"""Convert one verified formal Q2/Q3 CSV export into result2.xlsx/result3.xlsx."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from a_model.q2q3_xlsx import export_workbooks_fast


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Directory containing the three formal CSVs and metadata.json.")
    parser.add_argument("--output", type=Path, required=True, help="New directory for result2.xlsx and result3.xlsx.")
    parser.add_argument("--templates", type=Path, default=ROOT.parent / "A题" / "附件" / "附件3")
    args = parser.parse_args()
    result2, result3 = export_workbooks_fast(
        args.source, args.templates / "result2.xlsx", args.templates / "result3.xlsx", args.output
    )
    print(result2)
    print(result3)


if __name__ == "__main__":
    main()
