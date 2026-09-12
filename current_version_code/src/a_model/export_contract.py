"""Excel 与论文采样规则；不负责实际写文件。"""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite


DISPLAY_DECIMALS = 4
NUMBER_FORMAT = "0.0000"
Q1_END_S = 1800.0


def _decimal_grid(stop_tenths_cm: int) -> tuple[float, ...]:
    return tuple(index / 10.0 for index in range(stop_tenths_cm + 1))


FIXED_RADII_Q1_Q2_Q3_CM = _decimal_grid(20)
FIXED_RADII_Q4_CM = _decimal_grid(19)
Q4_SURFACE_COLUMN = "药材表面"
Q4_OUTSIDE_VALUE = None


@dataclass(frozen=True)
class ExportContract:
    case_id: str
    workbook: str
    interval_s: float
    first_time_s: float
    radii_cm: tuple[float, ...]
    sheets: tuple[str, ...]
    end_policy: str
    append_exact_endpoint: bool
    surface_column: str | None = None
    fixed_end_s: float | None = None

    def sample_times(self, end_time_s: float) -> tuple[float, ...]:
        if not isfinite(end_time_s) or end_time_s < self.first_time_s:
            raise ValueError("结束时间必须有限且不早于首个导出时刻。")
        if self.fixed_end_s is not None and not isclose(
            end_time_s, self.fixed_end_s, rel_tol=0.0, abs_tol=1.0e-9
        ):
            raise ValueError(f"{self.case_id} 的结束时间必须严格为 {self.fixed_end_s:g} s。")
        count = int((end_time_s + 1.0e-10) // self.interval_s)
        start_index = int(round(self.first_time_s / self.interval_s))
        values = [index * self.interval_s for index in range(start_index, count + 1)]
        if self.append_exact_endpoint and (
            not values or not isclose(values[-1], end_time_s, rel_tol=0.0, abs_tol=1.0e-9)
        ):
            values.append(end_time_s)
        return tuple(values)


EXPORTS: dict[str, ExportContract] = {
    "q1": ExportContract(
        "q1", "result1.xlsx", 1.0, 1.0, FIXED_RADII_Q1_Q2_Q3_CM,
        ("温度", "水分浓度"), "fixed_1800_s", False, fixed_end_s=Q1_END_S,
    ),
    "q2": ExportContract(
        "q2", "result2.xlsx", 1.0, 1.0, FIXED_RADII_Q1_Q2_Q3_CM,
        ("温度", "水分浓度"), "through_q3_event", True,
    ),
    "q3": ExportContract(
        "q3", "result3.xlsx", 60.0, 60.0, FIXED_RADII_Q1_Q2_Q3_CM,
        ("Sheet1",), "through_q3_event", True,
    ),
    "q4": ExportContract(
        "q4", "result4.xlsx", 60.0, 60.0, FIXED_RADII_Q4_CM,
        ("Sheet1",), "through_q4_event", True, Q4_SURFACE_COLUMN,
    ),
}


def q4_fixed_position_is_inside(
    radius_cm: float, current_radius_m: float, tolerance_m: float = 1.0e-10
) -> bool:
    if not all(isfinite(value) for value in (radius_cm, current_radius_m, tolerance_m)):
        raise ValueError("固定位置、当前半径和几何容差必须为有限数。")
    if radius_cm < 0.0 or current_radius_m <= 0.0 or tolerance_m < 0.0:
        raise ValueError("固定位置与容差必须非负，当前半径必须为正。")
    return radius_cm * 1.0e-2 <= current_radius_m + tolerance_m
