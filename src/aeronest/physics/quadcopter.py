"""DJI Mini-class quadcopter physics and battery estimates."""

from dataclasses import dataclass
from math import pi, sqrt


@dataclass(frozen=True)
class QuadcopterParameters:
    """Configuration for a small inspection quadcopter."""

    mass_kg: float = 0.30
    n_rotors: int = 4
    rotor_diameter_m: float = 0.12
    battery_voltage_nominal_v: float = 7.38
    battery_capacity_ah: float = 2.45
    usable_fraction: float = 0.80
    eta_motor_prop: float = 0.50
    avionics_power_w: float = 4.0
    payload_power_w: float = 2.0
    maneuver_factor: float = 1.25
    reserve_fraction: float = 0.35
    gravity_m_s2: float = 9.80665

    def __post_init__(self) -> None:
        if self.n_rotors <= 0:
            raise ValueError("n_rotors must be positive")
        for field_name in (
            "mass_kg",
            "rotor_diameter_m",
            "battery_voltage_nominal_v",
            "battery_capacity_ah",
            "avionics_power_w",
            "payload_power_w",
            "maneuver_factor",
            "gravity_m_s2",
        ):
            if getattr(self, field_name) <= 0.0:
                raise ValueError(f"{field_name} must be positive")
        for field_name in ("usable_fraction", "eta_motor_prop", "reserve_fraction"):
            value = getattr(self, field_name)
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{field_name} must be greater than 0 and at most 1")


@dataclass(frozen=True)
class QuadcopterModel:
    """Hover, flight-power, and battery model for Phase 2."""

    params: QuadcopterParameters
    air_density_kg_m3: float = 1.20

    @property
    def weight_n(self) -> float:
        return self.params.mass_kg * self.params.gravity_m_s2

    @property
    def rotor_disk_area_m2(self) -> float:
        radius_m = self.params.rotor_diameter_m / 2.0
        return self.params.n_rotors * pi * radius_m**2

    def ideal_hover_power_w(self) -> float:
        """Return ideal induced hover power in watts."""
        return self.weight_n**1.5 / sqrt(
            2.0 * self.air_density_kg_m3 * self.rotor_disk_area_m2
        )

    def electrical_hover_power_w(self) -> float:
        """Return electrical hover power including avionics."""
        return (
            self.ideal_hover_power_w() / self.params.eta_motor_prop
            + self.params.avionics_power_w
        )

    def flight_propulsion_power_w(self) -> float:
        """Return maneuver-adjusted hover power before payload draw."""
        return self.electrical_hover_power_w() * self.params.maneuver_factor

    def flight_power_w(self) -> float:
        """Return total flight power including payload."""
        return self.flight_propulsion_power_w() + self.params.payload_power_w

    def battery_total_wh(self) -> float:
        return self.params.battery_voltage_nominal_v * self.params.battery_capacity_ah

    def battery_usable_wh(self) -> float:
        return self.battery_total_wh() * self.params.usable_fraction

    def endurance_min(self) -> float:
        return 60.0 * self.battery_usable_wh() / self.flight_power_w()

    def reserve_aware_inspection_min(self) -> float:
        return self.endurance_min() * (1.0 - self.params.reserve_fraction)
