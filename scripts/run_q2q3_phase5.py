"""Bounded Q2/Q3 candidate scan or streaming CSV export (no Excel)."""

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from a_model.fvm import RadialGrid
from a_model.inputs import load_environment
from a_model.metadata import build_run_metadata
from a_model.parameters import get_case_parameters
from a_model.q2q3_phase5 import maximum_location, project, run_streaming
from a_model.solver import CoupledRadialSolver


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cells", type=int, default=20)
    parser.add_argument("--dt", type=float, default=60)
    parser.add_argument("--end", type=float, default=3600)
    parser.add_argument("--event-tolerance", type=float, default=.01)
    parser.add_argument("--mode", choices=("scan", "export"), default="scan")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    environment = load_environment()
    solver = CoupledRadialSolver(get_case_parameters("q2q3"), RadialGrid(args.cells), environment, .02)
    sources = sorted((ROOT / "src" / "a_model").glob("*.py")) + [Path(__file__).resolve()]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    metadata = build_run_metadata("q2q3", [environment.trace], {
        "cells": args.cells, "dt_s": args.dt, "end_time_s": args.end,
        "mode": args.mode, "event_tolerance_s": args.event_tolerance,
        "solver_options": asdict(solver.options), "convergence_verified": False,
        "sampling": "1 s temperature/moisture, 60 s moisture, plus event" if args.mode == "export" else "no periodic samples",
    })
    metadata.update(source_git_commit=commit, source_worktree_status=status, source_file_sha256=hashes,
                    result_classification="candidate_only", command=sys.argv)
    streams = []
    writers = []
    started = perf_counter()
    try:
        if args.mode == "export":
            for name in ("result2_temperature.csv", "result2_moisture.csv", "result3_moisture.csv"):
                stream = (args.output / name).open("w", newline="", encoding="utf-8")
                streams.append(stream)
                writer = csv.writer(stream)
                writer.writerow(("time_s", *(f"r_{i/10:g}_cm" for i in range(21))))
                writers.append(writer)
        def sample(state, terminal):
            if not writers or state.time_s == 0:
                return
            temperature, moisture = project(solver, state)
            writers[0].writerow((state.time_s, *temperature))
            writers[1].writerow((state.time_s, *moisture))
            if terminal or state.time_s % 60 == 0:
                writers[2].writerow((state.time_s, *moisture))
        result = run_streaming(solver, end_time_s=args.end, dt_s=args.dt,
                               sample_interval_s=1 if args.mode == "export" else None,
                               on_sample=sample, event_tolerance_s=args.event_tolerance)
        metadata["result"] = {k: asdict(v) if k in ("event", "state") else v for k, v in result.items()}
        final_state = result["event"].state if "event" in result else result["state"]
        metadata["maximum_location"] = maximum_location(solver, final_state)
    except Exception as exc:
        metadata["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        for stream in streams:
            stream.close()
        metadata["elapsed_wall_s"] = perf_counter() - started
        metadata["source_files_unchanged"] = all(hashlib.sha256(p.read_bytes()).hexdigest() == hashes[str(p.relative_to(ROOT))] for p in sources)
        (args.output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": result["status"], "steps": result["steps"], "elapsed_wall_s": metadata["elapsed_wall_s"]}))


if __name__ == "__main__":
    main()
