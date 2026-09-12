"""Q1、Q2/Q3 与 Q4 的显式参数包。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import exp, isfinite
from typing import Callable

from .conventions import (
    HEAT_TRANSFER_COEFFICIENT,
    MASS_TRANSFER_COEFFICIENT,
    to_kelvin,
)


ScalarLaw = Callable[[float], float]
DiffusivityLaw = Callable[[float, float], float]


def _require_state(moisture: float, temperature_c: float | None = None) -> None:
    if not isfinite(moisture) or moisture < 0.0:
        raise ValueError("干基含水率 C 必须是有限非负数。")
    if temperature_c is not None:
        if not isfinite(temperature_c) or to_kelvin(temperature_c) <= 0.0:
            raise ValueError("温度必须有限且高于绝对零度。")


def _q1_density(_: float) -> float:
    _require_state(_)
    return PROPERTY_COEFFICIENTS["q1"].rho_intercept


def _q1_heat_capacity(_: float) -> float:
    _require_state(_)
    return PROPERTY_COEFFICIENTS["q1"].cp_intercept


def _q1_conductivity(_: float) -> float:
    _require_state(_)
    return PROPERTY_COEFFICIENTS["q1"].k_intercept


def _q1_diffusivity(moisture: float, temperature_c: float) -> float:
    _require_state(moisture, temperature_c)
    coefficients = PROPERTY_COEFFICIENTS["q1"]
    effective_moisture = max(moisture, coefficients.diffusivity_moisture_floor)
    return coefficients.diffusivity_prefactor * exp(
        -coefficients.diffusivity_moisture_activation / effective_moisture
    )


def _q2_density(moisture: float) -> float:
    _require_state(moisture)
    coefficients = PROPERTY_COEFFICIENTS["q2q3"]
    return coefficients.rho_intercept + coefficients.rho_linear_c * moisture


def _q2_heat_capacity(moisture: float) -> float:
    _require_state(moisture)
    coefficients = PROPERTY_COEFFICIENTS["q2q3"]
    return coefficients.cp_intercept + coefficients.cp_c_over_one_plus_c * moisture / (1.0 + moisture)


def _q2_conductivity(moisture: float) -> float:
    _require_state(moisture)
    coefficients = PROPERTY_COEFFICIENTS["q2q3"]
    return coefficients.k_intercept + coefficients.k_c_over_one_plus_c * moisture / (1.0 + moisture)


def _q2_diffusivity(moisture: float, temperature_c: float) -> float:
    _require_state(moisture, temperature_c)
    coefficients = PROPERTY_COEFFICIENTS["q2q3"]
    effective_moisture = max(moisture, coefficients.diffusivity_moisture_floor)
    return coefficients.diffusivity_prefactor * exp(
        -coefficients.diffusivity_moisture_activation / effective_moisture
    ) * exp(-coefficients.diffusivity_temperature_activation_K / to_kelvin(temperature_c))


def _q4_density(moisture: float) -> float:
    _require_state(moisture)
    coefficients = PROPERTY_COEFFICIENTS["q4"]
    return coefficients.rho_intercept + coefficients.rho_linear_c * moisture


def _q4_heat_capacity(moisture: float) -> float:
    _require_state(moisture)
    coefficients = PROPERTY_COEFFICIENTS["q4"]
    return coefficients.cp_intercept + coefficients.cp_c_over_one_plus_c * moisture / (1.0 + moisture)


def _q4_conductivity(moisture: float) -> float:
    _require_state(moisture)
    coefficients = PROPERTY_COEFFICIENTS["q4"]
    return coefficients.k_intercept + coefficients.k_c_over_one_plus_c * moisture / (1.0 + moisture)


def _q4_diffusivity(moisture: float, temperature_c: float) -> float:
    _require_state(moisture, temperature_c)
    coefficients = PROPERTY_COEFFICIENTS["q4"]
    effective_moisture = max(moisture, coefficients.diffusivity_moisture_floor)
    return coefficients.diffusivity_prefactor * exp(
        -coefficients.diffusivity_moisture_activation / effective_moisture
    ) * exp(-coefficients.diffusivity_temperature_activation_K / to_kelvin(temperature_c))


@dataclass(frozen=True)
class CaseParameters:
    case_id: str
    source_appendix: str
    geometry: str
    density_parameter: ScalarLaw
    heat_capacity: ScalarLaw
    conductivity: ScalarLaw
    diffusivity: DiffusivityLaw
    heat_transfer_coefficient: float = HEAT_TRANSFER_COEFFICIENT
    mass_transfer_coefficient: float = MASS_TRANSFER_COEFFICIENT
    latent_heat_enabled: bool = False

    def volumetric_heat_storage(self, moisture: float) -> float:
        """V4 effective storage b=rho_g*cp; rho_g is not rho_d by default."""

        return self.density_parameter(moisture) * self.heat_capacity(moisture)


CASES: dict[str, CaseParameters] = {
    "q1": CaseParameters(
        "q1", "附录2", "fixed_radius", _q1_density, _q1_heat_capacity,
        _q1_conductivity, _q1_diffusivity,
    ),
    "q2q3": CaseParameters(
        "q2q3", "附录3", "fixed_radius", _q2_density, _q2_heat_capacity,
        _q2_conductivity, _q2_diffusivity,
    ),
    "q4": CaseParameters(
        "q4", "附录4", "proportional_radial_shrinkage", _q4_density,
        _q4_heat_capacity, _q4_conductivity, _q4_diffusivity,
    ),
}


@dataclass(frozen=True)
class PropertyCoefficients:
    rho_intercept: float
    rho_linear_c: float
    cp_intercept: float
    cp_c_over_one_plus_c: float
    k_intercept: float
    k_c_over_one_plus_c: float
    diffusivity_prefactor: float
    diffusivity_moisture_activation: float
    diffusivity_temperature_activation_K: float
    diffusivity_moisture_floor: float = 1.0e-8


PROPERTY_COEFFICIENTS: dict[str, PropertyCoefficients] = {
    "q1": PropertyCoefficients(820.0, 0.0, 2600.0, 0.0, 0.36, 0.0, 7.0e-9, 0.89, 0.0),
    "q2q3": PropertyCoefficients(650.0, 128.0, 1450.0, 2736.0, 0.21, 0.38, 2.4e-3, 0.45, 3850.0),
    "q4": PropertyCoefficients(760.0, 90.0, 1850.0, 2150.0, 0.12, 0.20, 4.2e-4, 0.30, 3850.0),
}


def parameter_snapshot(case_id: str) -> dict[str, object]:
    """Serialize the same coefficients exercised by the executable property laws."""

    parameters = get_case_parameters(case_id)
    model_ids = {
        "rho_g": "constant" if parameters.case_id == "q1" else "linear_in_C",
        "cp": "constant" if parameters.case_id == "q1" else "intercept_plus_C_over_1_plus_C",
        "k": "constant" if parameters.case_id == "q1" else "intercept_plus_C_over_1_plus_C",
        "D": "moisture_exponential" if parameters.case_id == "q1" else "moisture_temperature_exponential",
    }
    return {"model_ids": model_ids, "coefficients": asdict(PROPERTY_COEFFICIENTS[parameters.case_id])}


def get_case_parameters(case_id: str) -> CaseParameters:
    try:
        return CASES[case_id.lower()]
    except KeyError as exc:
        raise KeyError(f"未知工况 {case_id!r}；可用值为 {tuple(CASES)}。") from exc
