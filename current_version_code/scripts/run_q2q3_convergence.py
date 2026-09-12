"""Resume completed Q2/Q3 convergence cases and report measured differences."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from a_model.q2q3_convergence import (case_label, compare_cases, matrix_report, run_case,
                                     schedule_case_label, sha256, validated_summary)
from a_model.time_steps import v4_time_step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cells", type=int, nargs="+", default=[20, 40, 80])
    parser.add_argument("--dt", type=float, nargs="+", default=[60, 30, 15])
    parser.add_argument("--time-level", choices=("P0", "P1", "P2"), nargs="+",
                        help="Run nested V4 two-stage BE schedules instead of fixed --dt.")
    parser.add_argument("--time-switch", type=float, default=14400)
    parser.add_argument("--end", type=float, default=1200000)
    parser.add_argument("--sample", type=float, default=60)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--grading-exponent", type=float, default=1.0)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (len(set(args.cells)) != len(args.cells) or len(set(args.dt)) != len(args.dt)
            or args.time_level and len(set(args.time_level)) != len(args.time_level)):
        parser.error("Duplicate cases are not allowed.")
    relevant = ("__init__.py", "conventions.py", "parameters.py", "inputs.py", "metadata.py", "fvm.py",
                "solver.py", "q2q3_phase5.py", "q2q3_convergence.py", "time_steps.py")
    paths = [ROOT / "src" / "a_model" / name for name in relevant] + [Path(__file__).resolve()]
    provenance = {"source_file_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in paths},
                  "source_git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  "source_worktree_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True),
                  "command": sys.argv}
    if not args.report_only:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            tasks = {}
            requested = ([v4_time_step(level, args.time_switch) for level in args.time_level]
                         if args.time_level else args.dt)
            for item in requested:
                for n in args.cells:
                    schedule = item if args.time_level else None
                    dt = schedule.late_dt_s if schedule is not None else item
                    label = (schedule_case_label(n, schedule, args.grading_exponent)
                             if schedule is not None else case_label(n, dt, args.grading_exponent))
                    directory = args.output / label
                    summary_path = directory / "summary.json"
                    reused = False
                    if summary_path.exists():
                        try:
                            reused = bool(json.loads(summary_path.read_text(encoding="utf-8")).get("complete"))
                        except (OSError, json.JSONDecodeError):
                            pass
                    future = pool.submit(
                         run_case,
                         directory,
                         n,
                         dt,
                         args.end,
                         args.sample,
                         provenance,
                         args.grading_exponent,
                         schedule,
                    )
                    tasks[future] = (label, reused)
            for future in as_completed(tasks):
                result = future.result()
                label, reused = tasks[future]
                print(json.dumps({"case": label,
                                  "progress": "reused_completed" if reused else "computed",
                                  "status": result["result"]["status"],
                                  "last_time_s": result["final_snapshot"]["time_s"],
                                  "elapsed_s": result["elapsed_wall_s"]}), flush=True)
    if args.time_level:
        schedules = [v4_time_step(level, args.time_switch) for level in args.time_level]
        comparisons = []
        completed = 0
        for n in args.cells:
            directories = [args.output / schedule_case_label(n, schedule, args.grading_exponent)
                           for schedule in schedules]
            available = []
            for directory, schedule in zip(directories, schedules):
                if (directory / "summary.json").is_file():
                    validated_summary(
                        directory, expected_cells=n, expected_dt_s=schedule.late_dt_s,
                        expected_grading_exponent=args.grading_exponent,
                        expected_source_hashes=provenance["source_file_sha256"],
                    )
                    available.append(directory)
                    completed += 1
            for left, right in zip(directories, directories[1:]):
                if left in available and right in available:
                    comparison = compare_cases(left, right)
                    comparison["cells"] = n
                    comparisons.append(comparison)
        final_pairs = [item for item in comparisons
                       if item["right"] == schedule_case_label(
                           item["cells"], schedules[-1], args.grading_exponent)]
        requested_cases = len(args.cells) * len(schedules)
        report = {
            "result_classification": "candidate_only",
            "convergence_verified": False,
            "time_gate_passed": completed == requested_cases and len(final_pairs) == len(args.cells)
                and all(all(item["starter_thresholds_pass"].values()) for item in final_pairs),
            "completed_cases": completed,
            "requested_cases": requested_cases,
            "time_levels": [schedule.as_dict() for schedule in schedules],
            "comparisons": comparisons,
        }
        report_path = args.output / "time_schedule_report.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
                               encoding="utf-8")
    else:
        report = matrix_report(args.output, args.cells, args.dt, args.grading_exponent,
                               expected_source_hashes=provenance["source_file_sha256"])
        report_path = args.output / "convergence_report.json"
    unchanged = all(sha256(p) == provenance["source_file_sha256"][str(p.relative_to(ROOT))] for p in paths)
    if not unchanged:
        raise RuntimeError("Relevant source changed during the matrix run; inspect provenance before reuse.")
    print(json.dumps({"completed_cases": report["completed_cases"], "requested_cases": report["requested_cases"],
                      "convergence_verified": False, "report": str(report_path)}), flush=True)


if __name__ == "__main__":
    main()
