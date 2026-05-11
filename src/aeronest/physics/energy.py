"""Energy models for AeroNest Phase 2 mission simulation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SolarParameters:
    panel_area_m2: float = 0.35
    panel_efficiency: float = 0.22
    irradiance_w_m2: float = 800.0
    sun_angle_factor: float = 0.75
    shading_factor: float = 0.75
    mppt_efficiency: float = 0.92

    def __post_init__(self) -> None:
        for field_name in ("panel_area_m2", "irradiance_w_m2"):
            if getattr(self, field_name) <= 0.0:
                raise ValueError(f"{field_name} must be positive")
        for field_name in (
            "panel_efficiency",
            "sun_angle_factor",
            "shading_factor",
            "mppt_efficiency",
        ):
            value = getattr(self, field_name)
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{field_name} must be greater than 0 and at most 1")


@dataclass(frozen=True)
class MothershipEnergyParameters:
    battery_capacity_wh: float = 150.0
    avionics_power_w: float = 8.0
    charge_power_w: float = 15.0
    charge_efficiency: float = 0.85
    fallback_propulsion_power_w: float = 55.0

    def __post_init__(self) -> None:
        for field_name in (
            "battery_capacity_wh",
            "avionics_power_w",
            "charge_power_w",
            "fallback_propulsion_power_w",
        ):
            if getattr(self, field_name) <= 0.0:
                raise ValueError(f"{field_name} must be positive")
        if not 0.0 < self.charge_efficiency <= 1.0:
            raise ValueError("charge_efficiency must be greater than 0 and at most 1")


def solar_power_w(params: SolarParameters) -> float:
    """Return panel power output in watts."""
    return (
        params.panel_efficiency
        * params.panel_area_m2
        * params.irradiance_w_m2
        * params.sun_angle_factor
        * params.shading_factor
        * params.mppt_efficiency
    )


def mothership_energy_rate_w(
    solar_power: float,
    propulsion_power_w: float,
    avionics_power_w: float,
    charge_power_w: float = 0.0,
) -> float:
    """Return mothership battery energy rate in watts."""
    return solar_power - propulsion_power_w - avionics_power_w - charge_power_w


def quad_flight_energy_rate_w(propulsion_power_w: float, payload_power_w: float) -> float:
    """Return quadcopter battery rate during flight in watts."""
    return -(propulsion_power_w + payload_power_w)


def quad_charge_energy_rate_w(charge_power_w: float, charge_efficiency: float) -> float:
    """Return quadcopter battery rate during charging in watts."""
    return charge_efficiency * charge_power_w
