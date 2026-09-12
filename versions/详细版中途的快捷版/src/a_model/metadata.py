"""为后续数值结果生成可序列化的追溯元数据。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import platform
from typing import Any, Iterable

from .conventions import (
    ASSUMPTIONS,
    CYLINDER_LENGTH_M,
    INITIAL_MOISTURE_DRY_BASIS,
    INITIAL_RADIUS_M,
    INITIAL_TEMPERATURE_C,
    MODEL_VERSION,
    TARGET_MAX_MOISTURE,
    UNITS,
)
from .inputs import SourceTrace
from .parameters import get_case_parameters, parameter_snapshot


DEFAULT_ENVIRONMENT_POLICY: dict[str, Any] = {
    "in_range": "piecewise_linear_0_to_14400_s",
    "after_range": {"temperature_C": 50.0, "equilibrium_moisture": 0.05},
}

DEFAULT_GEOMETRY_POLICIES: dict[str, dict[str, Any]] = {
    "q1": {"kind": "fixed_radius", "radius_m": INITIAL_RADIUS_M},
    "q2q3": {"kind": "fixed_radius", "radius_m": INITIAL_RADIUS_M},
    "q4": {
        "kind": "proportional_radial_shrinkage",
        "radius_in_range": "attachment2_piecewise_linear_0_to_259200_s",
        "radius_after_range_m": 0.01198,
    },
}

REQUIRED_SOURCE_FILENAMES: dict[str, frozenset[str]] = {
    "q1": frozenset({"附件1.xlsx"}),
    "q2q3": frozenset({"附件1.xlsx"}),
    "q4": frozenset({"附件1.xlsx", "附件2.xlsx"}),
}


def _installed_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "not-installed"


def build_run_metadata(
    case_id: str,
    sources: Iterable[SourceTrace],
    numerical_settings: dict[str, Any],
    *,
    created_at: datetime | None = None,
    input_policy: dict[str, Any] | None = None,
    geometry_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    parameters = get_case_parameters(case_id)
    source_list = list(sources)
    filenames = {source.filename for source in source_list}
    missing_sources = REQUIRED_SOURCE_FILENAMES[parameters.case_id] - filenames
    if missing_sources:
        raise ValueError(
            f"{parameters.case_id} 运行元数据缺少必需来源：{sorted(missing_sources)}。"
        )
    stamp = created_at or datetime.now(timezone.utc)
    if stamp.tzinfo is None:
        raise ValueError("created_at 必须包含时区。")
    return {
        "model_version": MODEL_VERSION,
        "case_id": parameters.case_id,
        "source_appendix": parameters.source_appendix,
        "geometry": deepcopy(
            geometry_policy
            if geometry_policy is not None
            else DEFAULT_GEOMETRY_POLICIES[parameters.case_id]
        ),
        "created_at_utc": stamp.astimezone(timezone.utc).isoformat(),
        "sources": [asdict(source) for source in source_list],
        "units": asdict(UNITS),
        "assumptions": dict(ASSUMPTIONS),
        "initial_conditions": {
            "temperature_C": INITIAL_TEMPERATURE_C,
            "moisture_dry_basis": INITIAL_MOISTURE_DRY_BASIS,
            "radius_m": INITIAL_RADIUS_M,
            "cylinder_length_m": CYLINDER_LENGTH_M,
        },
        "target": {"max_moisture_dry_basis": TARGET_MAX_MOISTURE},
        "property_model": {
            "effective_heat_storage": "b(C)=rho_g(C)*cp(C)",
            "rho_g_is_not_rho_d_by_default": True,
            **parameter_snapshot(parameters.case_id),
        },
        "boundary_coefficients": {
            "h_W_m2_K": parameters.heat_transfer_coefficient,
            "hm_m_s": parameters.mass_transfer_coefficient,
            "carried_forward_from": "附录2" if parameters.case_id != "q1" else "题给附录2",
        },
        "input_policy": deepcopy(
            input_policy if input_policy is not None else DEFAULT_ENVIRONMENT_POLICY
        ),
        "runtime": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "dependencies": {"openpyxl": _installed_version("openpyxl")},
        },
        "latent_heat_enabled": parameters.latent_heat_enabled,
        "numerical_settings": deepcopy(numerical_settings),
    }
