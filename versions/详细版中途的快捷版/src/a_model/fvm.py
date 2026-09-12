"""Cell-centred finite volumes for the V4 radial cylinder model.

The unknowns are radial-volume-weighted cell averages on the fixed material
coordinate ``xi in [0, 1]``.  This module only contains spatial algebra; time
integration and nonlinear iteration live in :mod:`a_model.solver`.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Integral
from typing import Sequence


class DiscretizationError(ValueError):
    """A grid, coefficient, or linear system violates the FVM assumptions."""


@dataclass(frozen=True)
class RadialGrid:
    """Uniform, cell-centred grid in the dimensionless radial coordinate."""

    cell_count: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.cell_count, bool)
            or not isinstance(self.cell_count, Integral)
            or self.cell_count < 2
        ):
            raise DiscretizationError("The radial grid needs at least two cells.")

    @property
    def spacing(self) -> float:
        return 1.0 / self.cell_count

    @property
    def faces(self) -> tuple[float, ...]:
        dx = self.spacing
        return tuple(index * dx for index in range(self.cell_count + 1))

    @property
    def centers(self) -> tuple[float, ...]:
        dx = self.spacing
        return tuple((index + 0.5) * dx for index in range(self.cell_count))

    @property
    def weights(self) -> tuple[float, ...]:
        faces = self.faces
        return tuple(right * right - left * left for left, right in zip(faces, faces[1:]))


@dataclass(frozen=True)
class TridiagonalSystem:
    """Three diagonals and a right-hand side, all stored without padding."""

    lower: tuple[float, ...]
    diagonal: tuple[float, ...]
    upper: tuple[float, ...]
    right_hand_side: tuple[float, ...]

    def __post_init__(self) -> None:
        size = len(self.diagonal)
        if size < 1 or len(self.right_hand_side) != size:
            raise DiscretizationError("Diagonal and right-hand side sizes do not match.")
        if len(self.lower) != size - 1 or len(self.upper) != size - 1:
            raise DiscretizationError("Off-diagonal sizes must be one less than the diagonal.")
        values = (*self.lower, *self.diagonal, *self.upper, *self.right_hand_side)
        if any(not isfinite(value) for value in values):
            raise DiscretizationError("The tridiagonal system contains a non-finite value.")


@dataclass(frozen=True)
class FrozenOperator:
    """Frozen positive coefficients used in one backward-Euler linear solve."""

    storage: tuple[float, ...]
    internal_conductances: tuple[float, ...]
    boundary_conductance: float


def distance_weighted_harmonic(
    left_value: float,
    right_value: float,
    left_distance: float,
    right_distance: float,
) -> float:
    """Return the interface coefficient that preserves series resistance."""

    values = (left_value, right_value, left_distance, right_distance)
    if any(not isfinite(value) for value in values):
        raise DiscretizationError("Harmonic-mean values and distances must be finite.")
    if left_value < 0.0 or right_value < 0.0:
        raise DiscretizationError("Harmonic-mean transport values must be non-negative.")
    if left_distance <= 0.0 or right_distance <= 0.0:
        raise DiscretizationError("Harmonic-mean distances must be positive.")
    if left_value == 0.0 or right_value == 0.0:
        return 0.0
    return (left_distance + right_distance) / (
        left_distance / left_value + right_distance / right_value
    )


def internal_conductances(
    grid: RadialGrid, diffusivities: Sequence[float], radius_m: float
) -> tuple[float, ...]:
    """Compute the shared normalized conductance at every internal face."""

    if len(diffusivities) != grid.cell_count:
        raise DiscretizationError("One transport coefficient is required per cell.")
    if not isfinite(radius_m) or radius_m <= 0.0:
        raise DiscretizationError("Radius must be finite and positive.")
    if any(not isfinite(value) or value < 0.0 for value in diffusivities):
        raise DiscretizationError("Transport coefficients must be finite and non-negative.")

    dx = grid.spacing
    result: list[float] = []
    for face_index in range(1, grid.cell_count):
        face = face_index * dx
        coefficient = distance_weighted_harmonic(
            diffusivities[face_index - 1], diffusivities[face_index], dx / 2.0, dx / 2.0
        )
        result.append(2.0 * face * coefficient / (radius_m * radius_m * dx))
    return tuple(result)


def surface_conductance(
    grid: RadialGrid,
    radius_m: float,
    boundary_transport: float,
    transfer_coefficient: float,
) -> float:
    """Return ``g_s`` after eliminating the Robin boundary face value."""

    values = (radius_m, boundary_transport, transfer_coefficient)
    if any(not isfinite(value) for value in values):
        raise DiscretizationError("Radius and Robin coefficients must be finite.")
    if radius_m <= 0.0 or transfer_coefficient <= 0.0 or boundary_transport < 0.0:
        raise DiscretizationError(
            "Radius and film coefficient must be positive; boundary transport may be zero."
        )
    if boundary_transport == 0.0:
        return 0.0
    delta_m = radius_m * (1.0 - grid.centers[-1])
    return (2.0 / radius_m) / (delta_m / boundary_transport + 1.0 / transfer_coefficient)


def reconstruct_surface(
    grid: RadialGrid,
    last_cell_value: float,
    environment_value: float,
    radius_m: float,
    boundary_transport: float,
    transfer_coefficient: float,
) -> float:
    """Reconstruct the Robin face value including the physical half-cell resistance."""

    if not isfinite(last_cell_value) or not isfinite(environment_value):
        raise DiscretizationError("Cell and environment values must be finite.")
    if not isfinite(radius_m) or radius_m <= 0.0:
        raise DiscretizationError("Radius must be finite and positive.")
    if not isfinite(boundary_transport) or boundary_transport < 0.0:
        raise DiscretizationError("Boundary transport coefficient must be finite and non-negative.")
    if not isfinite(transfer_coefficient) or transfer_coefficient <= 0.0:
        raise DiscretizationError("Robin transfer coefficient must be finite and positive.")
    if boundary_transport == 0.0:
        return environment_value
    delta_m = radius_m * (1.0 - grid.centers[-1])
    return (
        boundary_transport * last_cell_value
        + transfer_coefficient * delta_m * environment_value
    ) / (boundary_transport + transfer_coefficient * delta_m)


def reconstruct_center(cell_values: Sequence[float]) -> float:
    """Even-quadratic centre reconstruction for radial cell averages (V4 eq. 42)."""

    if len(cell_values) < 2:
        raise DiscretizationError("Centre reconstruction needs the first two cell averages.")
    if not isfinite(cell_values[0]) or not isfinite(cell_values[1]):
        raise DiscretizationError("Cell values must be finite.")
    return (5.0 * cell_values[0] - cell_values[1]) / 4.0


def freeze_operator(
    grid: RadialGrid,
    storage_coefficients: Sequence[float],
    transport_coefficients: Sequence[float],
    radius_m: float,
    boundary_transport: float,
    transfer_coefficient: float,
) -> FrozenOperator:
    """Build all positive row coefficients shared by assembly and residual checks."""

    if len(storage_coefficients) != grid.cell_count:
        raise DiscretizationError("One storage coefficient is required per cell.")
    if any(not isfinite(value) or value <= 0.0 for value in storage_coefficients):
        raise DiscretizationError("Storage coefficients must be finite and positive.")
    return FrozenOperator(
        tuple(float(value) for value in storage_coefficients),
        internal_conductances(grid, transport_coefficients, radius_m),
        surface_conductance(grid, radius_m, boundary_transport, transfer_coefficient),
    )


def assemble_backward_euler(
    grid: RadialGrid,
    operator: FrozenOperator,
    previous_values: Sequence[float],
    dt_s: float,
    environment_value: float,
) -> TridiagonalSystem:
    """Assemble V4 equations (38)--(41) for one frozen Picard iterate."""

    count = grid.cell_count
    if len(previous_values) != count or len(operator.storage) != count:
        raise DiscretizationError("State size does not match the grid.")
    if len(operator.internal_conductances) != count - 1:
        raise DiscretizationError("The frozen operator has the wrong internal-face count.")
    if not isfinite(dt_s) or dt_s <= 0.0 or not isfinite(environment_value):
        raise DiscretizationError("Time step must be positive and environment value finite.")
    if any(not isfinite(value) for value in previous_values):
        raise DiscretizationError("Previous state contains a non-finite value.")

    accumulation = tuple(
        storage * weight / dt_s for storage, weight in zip(operator.storage, grid.weights)
    )
    lower = tuple(-value for value in operator.internal_conductances)
    upper = lower
    diagonal: list[float] = []
    right_hand_side: list[float] = []
    for index in range(count):
        left = operator.internal_conductances[index - 1] if index > 0 else 0.0
        right = operator.internal_conductances[index] if index < count - 1 else 0.0
        boundary = operator.boundary_conductance if index == count - 1 else 0.0
        diagonal.append(accumulation[index] + left + right + boundary)
        rhs = accumulation[index] * previous_values[index]
        if index == count - 1:
            rhs += boundary * environment_value
        right_hand_side.append(rhs)
    return TridiagonalSystem(lower, tuple(diagonal), upper, tuple(right_hand_side))


def solve_tridiagonal(system: TridiagonalSystem) -> tuple[float, ...]:
    """Solve a strictly positive FVM tridiagonal system with Thomas elimination."""

    size = len(system.diagonal)
    diagonal = list(system.diagonal)
    rhs = list(system.right_hand_side)
    scale = max((abs(value) for value in diagonal), default=1.0)
    pivot_floor = max(scale * 1.0e-14, 1.0e-300)
    if diagonal[0] <= pivot_floor:
        raise DiscretizationError("The tridiagonal system has a non-positive first pivot.")
    for index in range(1, size):
        multiplier = system.lower[index - 1] / diagonal[index - 1]
        diagonal[index] -= multiplier * system.upper[index - 1]
        rhs[index] -= multiplier * rhs[index - 1]
        if diagonal[index] <= pivot_floor or not isfinite(diagonal[index]):
            raise DiscretizationError("The tridiagonal system lost a positive pivot.")
    result = [0.0] * size
    result[-1] = rhs[-1] / diagonal[-1]
    for index in range(size - 2, -1, -1):
        result[index] = (
            rhs[index] - system.upper[index] * result[index + 1]
        ) / diagonal[index]
    if any(not isfinite(value) for value in result):
        raise DiscretizationError("The tridiagonal solve produced a non-finite state.")
    return tuple(result)


def nonlinear_row_residuals(
    grid: RadialGrid,
    operator: FrozenOperator,
    previous_values: Sequence[float],
    new_values: Sequence[float],
    dt_s: float,
    environment_value: float,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return dimensional row residuals and locally scaled absolute residuals."""

    count = grid.cell_count
    if len(previous_values) != count or len(new_values) != count:
        raise DiscretizationError("State size does not match the grid.")
    accumulation = tuple(
        storage * weight / dt_s for storage, weight in zip(operator.storage, grid.weights)
    )
    residuals: list[float] = []
    scaled: list[float] = []
    for index in range(count):
        storage_term = accumulation[index] * (new_values[index] - previous_values[index])
        flux_terms: list[float] = []
        algebraic_scale = accumulation[index] * (
            abs(new_values[index]) + abs(previous_values[index])
        )
        if index > 0:
            conductance = operator.internal_conductances[index - 1]
            flux_terms.append(
                conductance * (new_values[index] - new_values[index - 1])
            )
            algebraic_scale += conductance * (
                abs(new_values[index]) + abs(new_values[index - 1])
            )
        if index < count - 1:
            conductance = operator.internal_conductances[index]
            flux_terms.append(
                conductance * (new_values[index] - new_values[index + 1])
            )
            algebraic_scale += conductance * (
                abs(new_values[index]) + abs(new_values[index + 1])
            )
        else:
            flux_terms.append(
                operator.boundary_conductance * (new_values[index] - environment_value)
            )
            algebraic_scale += operator.boundary_conductance * (
                abs(new_values[index]) + abs(environment_value)
            )
        residual = storage_term + sum(flux_terms)
        physical_scale = abs(storage_term) + sum(abs(term) for term in flux_terms)
        # ``algebraic_scale`` is the row magnitude before cancellation.  It is
        # essential at unchanged inner cells, where the physical increment and
        # flux are both close to zero and a relative residual based only on
        # their difference would amplify floating-point roundoff.
        scale = max(physical_scale, algebraic_scale, 1.0e-300)
        residuals.append(residual)
        scaled.append(abs(residual) / scale)
    return tuple(residuals), tuple(scaled)


def sample_reconstructed(
    grid: RadialGrid,
    cell_values: Sequence[float],
    xi: float,
    surface_value: float,
) -> float:
    """Sample the documented piecewise-linear output reconstruction."""

    if len(cell_values) != grid.cell_count:
        raise DiscretizationError("State size does not match the grid.")
    if not isfinite(xi) or xi < 0.0 or xi > 1.0:
        raise DiscretizationError("xi must lie in [0, 1].")
    if xi == 0.0:
        return reconstruct_center(cell_values)
    if xi == 1.0:
        return surface_value
    points = (0.0, *grid.centers, 1.0)
    values = (reconstruct_center(cell_values), *cell_values, surface_value)
    for left_index, right_point in enumerate(points[1:]):
        if xi <= right_point:
            left_point = points[left_index]
            fraction = (xi - left_point) / (right_point - left_point)
            return values[left_index] + fraction * (values[left_index + 1] - values[left_index])
    raise AssertionError("unreachable")


def maximum_reconstructed(
    grid: RadialGrid, cell_values: Sequence[float], surface_value: float
) -> float:
    """Maximum of the current piecewise-linear reconstruction, including both boundaries."""

    return max(reconstruct_center(cell_values), *cell_values, surface_value)
