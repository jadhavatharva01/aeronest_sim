"""Fixed-wing slow-flight physics for the AeroNest mothership."""

from dataclasses import dataclass
from math import pi, sqrt

import numpy as np

from aeronest.physics.atmosphere import Atmosphere


@dataclass(frozen=True)
class FixedWingParameters:
    """Configuration for the stacked-wing fixed-wing mothership."""

    n_wings: int = 3
    span_m: float = 1.5
    chord_m: float = 0.18
    interference_factor: float = 0.75
    mass_kg: float = 2.2
    cl_max: float = 1.6
    cd0: float = 0.055
    oswald_efficiency: float = 0.78
    propeller_efficiency: float = 0.62
    gravity_m_s2: float = 9.80665

    def __post_init__(self) -> None:
        if self.n_wings <= 0:
            raise ValueError("n_wings must be positive")
        for field_name in (
            "span_m",
            "chord_m",
            "mass_kg",
            "cl_max",
            "cd0",
            "gravity_m_s2",
        ):
            if getattr(self, field_name) <= 0.0:
                raise ValueError(f"{field_name} must be positive")
        for field_name in (
            "interference_factor",
            "oswald_efficiency",
            "propeller_efficiency",
        ):
            value = getattr(self, field_name)
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{field_name} must be greater than 0 and at most 1")


@dataclass(frozen=True)
class FixedWingModel:
    """Steady fixed-wing model for slow-flight feasibility checks."""

    params: FixedWingParameters
    atmosphere: Atmosphere

    @property
    def effective_wing_area_m2(self) -> float:
        """Stacked effective wing area after interference losses."""
        return (
            self.params.n_wings
            * self.params.span_m
            * self.params.chord_m
            * self.params.interference_factor
        )

    @property
    def aspect_ratio(self) -> float:
        """Aspect ratio based on span and effective lifting area."""
        return self.params.span_m**2 / self.effective_wing_area_m2

    @property
    def weight_n(self) -> float:
        return self.params.mass_kg * self.params.gravity_m_s2

    def lift_n(self, speed_m_s: float, cl: float) -> float:
        """Return lift force in newtons."""
        return (
            self.atmosphere.dynamic_pressure(speed_m_s)
            * self.effective_wing_area_m2
            * cl
        )

    def drag_coefficient(self, cl: float) -> float:
        """Return drag coefficient from parasite and induced drag."""
        induced = cl**2 / (pi * self.params.oswald_efficiency * self.aspect_ratio)
        return self.params.cd0 + induced

    def drag_n(self, speed_m_s: float, cl: float) -> float:
        """Return drag force in newtons."""
        return (
            self.atmosphere.dynamic_pressure(speed_m_s)
            * self.effective_wing_area_m2
            * self.drag_coefficient(cl)
        )

    def stall_speed_m_s(self) -> float:
        """Return steady level-flight stall speed in meters per second."""
        denominator = (
            self.atmosphere.rho
            * self.effective_wing_area_m2
            * self.params.cl_max
        )
        return sqrt((2.0 * self.weight_n) / denominator)

    def recommended_docking_speed_m_s(self, factor: float = 1.25) -> float:
        """Return recommended docking speed from a stall margin factor."""
        if factor <= 0.0:
            raise ValueError("factor must be positive")
        return factor * self.stall_speed_m_s()

    def lift_coefficient_for_level_flight(self, speed_m_s: float) -> float:
        """Return the lift coefficient required for steady level flight."""
        if speed_m_s <= 0.0:
            raise ValueError("speed_m_s must be positive")
        q_s = self.atmosphere.dynamic_pressure(speed_m_s) * self.effective_wing_area_m2
        return self.weight_n / q_s

    def power_required_w(self, speed_m_s: float) -> float:
        """Return propulsive power required for steady level flight."""
        if speed_m_s <= 0.0:
            raise ValueError("speed_m_s must be positive")
        cl = self.lift_coefficient_for_level_flight(speed_m_s)
        drag = self.drag_n(speed_m_s, cl)
        return drag * speed_m_s / self.params.propeller_efficiency

    def performance_sweep(self, speeds_m_s: np.ndarray) -> dict[str, np.ndarray]:
        """Return lift, drag, CL, and power arrays over a speed sweep."""
        speeds = np.asarray(speeds_m_s, dtype=float)
        if np.any(speeds <= 0.0):
            raise ValueError("all speeds must be positive")

        cl = np.array([self.lift_coefficient_for_level_flight(v) for v in speeds])
        drag = np.array([self.drag_n(v, cl_i) for v, cl_i in zip(speeds, cl)])
        power = drag * speeds / self.params.propeller_efficiency
        lift_at_clmax = np.array([self.lift_n(v, self.params.cl_max) for v in speeds])

        return {
            "speed_m_s": speeds,
            "cl_level": cl,
            "drag_n": drag,
            "power_required_w": power,
            "lift_at_clmax_n": lift_at_clmax,
        }
