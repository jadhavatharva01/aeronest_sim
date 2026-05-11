"""Docking pad geometry and capture envelope."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DockingPadModel:
    """2D capture geometry for the mothership docking bay."""

    bay_length_m: float = 0.40
    bay_width_m: float = 0.30
    x_tolerance_m: float = 0.10
    z_tolerance_m: float = 0.05
    relative_velocity_tolerance_m_s: float = 0.25
    stable_capture_time_s: float = 1.0

    def __post_init__(self) -> None:
        for field_name in (
            "bay_length_m",
            "bay_width_m",
            "x_tolerance_m",
            "z_tolerance_m",
            "relative_velocity_tolerance_m_s",
            "stable_capture_time_s",
        ):
            if getattr(self, field_name) <= 0.0:
                raise ValueError(f"{field_name} must be positive")
