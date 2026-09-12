"""Reusable time-step schedules for the V4 Q2/Q3 calculations."""

from dataclasses import dataclass
from math import isfinite


V4_TIME_LEVELS = {
    "P0": (0.5, 1.0),
    "P1": (0.25, 0.5),
    "P2": (0.125, 0.25),
}


@dataclass(frozen=True)
class TwoStageTimeStep:
    """Use ``early_dt_s`` before one exact switch, then ``late_dt_s``."""

    level: str
    early_dt_s: float
    late_dt_s: float
    switch_time_s: float = 14400.0

    def __post_init__(self):
        values = (self.early_dt_s, self.late_dt_s, self.switch_time_s)
        if any(not isfinite(value) or value <= 0 for value in values):
            raise ValueError("Time-step sizes and switch time must be finite and positive.")

    def __call__(self, time_s: float) -> float:
        if not isfinite(time_s) or time_s < 0:
            raise ValueError("Time must be finite and non-negative.")
        return self.early_dt_s if time_s < self.switch_time_s else self.late_dt_s

    @property
    def breakpoints_s(self) -> tuple[float, ...]:
        return (self.switch_time_s,)

    @property
    def step_sizes_s(self) -> tuple[float, ...]:
        return self.early_dt_s, self.late_dt_s

    def as_dict(self) -> dict:
        return {
            "kind": "two_stage_backward_euler",
            "level": self.level,
            "early_dt_s": self.early_dt_s,
            "late_dt_s": self.late_dt_s,
            "switch_time_s": self.switch_time_s,
        }


def v4_time_step(level: str, switch_time_s: float = 14400.0) -> TwoStageTimeStep:
    """Return one of the three nested V4 schedules P0, P1, or P2."""
    try:
        early_dt_s, late_dt_s = V4_TIME_LEVELS[level.upper()]
    except (AttributeError, KeyError) as exc:
        raise ValueError("V4 time level must be P0, P1, or P2.") from exc
    return TwoStageTimeStep(level.upper(), early_dt_s, late_dt_s, switch_time_s)
