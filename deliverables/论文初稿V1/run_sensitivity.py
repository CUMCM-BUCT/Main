"""Run reproducible ±20% sensitivity cases for the paper figures.

The baseline physics, environment and radius inputs are imported from the
validated solver.  Only one parameter is perturbed at a time (OAT), which
makes the reported effects directly attributable to D, h or hm.
"""
from __future__ import annotations

import csv
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from a_model.inputs import load_environment, load_radius
from a_model.parameters import get_case_parameters
from a_model.fvm import RadialGrid
from a_model.q2q3_phase5 import run_streaming, surface, maximum
from a_model.solver import CoupledRadialSolver

OUT = Path(__file__).resolve().parent / "sensitivity"
OUT.mkdir(exist_ok=True)


def perturbed(case_id: str, name: str, factor: float):
    base = get_case_parameters(case_id)
    if name == "D":
        old = base.diffusivity
        law = lambda c, t: factor * old(c, t)
        return replace(base, diffusivity=law)
    if name == "h":
        return replace(base, heat_transfer_coefficient=base.heat_transfer_coefficient * factor)
    if name == "hm":
        return replace(base, mass_transfer_coefficient=base.mass_transfer_coefficient * factor)
    raise ValueError(name)


def one_run(case_id: str, name: str, factor: float, cells: int, dt: float, end: float):
    env = load_environment()
    radius = load_radius() if case_id == "q4" else 0.02
    grid = RadialGrid(cells, 2.0)
    solver = CoupledRadialSolver(perturbed(case_id, name, factor), grid, env, radius)
    result = run_streaming(solver, end_time_s=end, dt_s=dt, event_tolerance_s=0.1)
    if result["status"] == "event":
        state = result["event"].state
        tf = state.time_s
    else:
        state = result["state"]
        tf = None
    Ts, Cs = surface(solver, state)
    return {
        "case": case_id, "parameter": name, "factor": factor,
        "event_time_s": tf, "event_time_h": None if tf is None else tf / 3600.0,
        "center_temperature_C": state.temperatures_c[0], "surface_temperature_C": Ts,
        "center_moisture": state.moistures[0], "surface_moisture": Cs,
        "maximum_moisture": maximum(solver, state), "steps": result["steps"],
    }


def q1_snapshot(name: str, factor: float):
    env = load_environment()
    grid = RadialGrid(320, 2.0)
    solver = CoupledRadialSolver(perturbed("q2q3", name, factor), grid, env, 0.02)
    # Use the variable-property solver for a like-for-like short-time field.
    state = solver.initial_state()
    while state.time_s < 1800.0:
        state = solver.advance(state, min(3.75, 1800.0 - state.time_s)).state
    Ts, Cs = surface(solver, state)
    return {"case": "short", "parameter": name, "factor": factor,
            "event_time_s": None, "event_time_h": None,
            "center_temperature_C": state.temperatures_c[0], "surface_temperature_C": Ts,
            "center_moisture": state.moistures[0], "surface_moisture": Cs,
            "maximum_moisture": maximum(solver, state), "steps": None}


def write_csv(path: Path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    factors = (0.8, 1.0, 1.2)
    short = [q1_snapshot(p, f) for p in ("D", "h", "hm") for f in factors]
    write_csv(OUT / "short_q2q3_sensitivity.csv", short)
    q3 = [one_run("q2q3", p, f, cells=80, dt=60.0, end=400000.0)
          for p in ("D", "h", "hm") for f in factors]
    write_csv(OUT / "q3_event_sensitivity.csv", q3)
    q4 = [one_run("q4", p, f, cells=40, dt=60.0, end=300000.0)
          for p in ("D", "h", "hm") for f in factors]
    write_csv(OUT / "q4_event_sensitivity.csv", q4)
    (OUT / "README.json").write_text(json.dumps({
        "design": "one-at-a-time ±20% perturbation",
        "q1_like_short": "q2q3 variable-property model at 1800 s, N=320, dt=3.75 s",
        "q3": "N=80, dt=60 s candidate event runs",
        "q4": "N=40, dt=60 s candidate event runs",
        "parameters": ["D", "h", "hm"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("sensitivity complete", OUT)
