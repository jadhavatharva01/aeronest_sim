"""Phase 2 mission-level energy simulation."""

from dataclasses import dataclass

import numpy as np

from aeronest.control.state_machine import MissionState
from aeronest.physics.energy import (
    MothershipEnergyParameters,
    SolarParameters,
    mothership_energy_rate_w,
    quad_charge_energy_rate_w,
    quad_flight_energy_rate_w,
    solar_power_w,
)
from aeronest.physics.quadcopter import QuadcopterModel


@dataclass(frozen=True)
class MissionTiming:
    preflight_min: float = 1.0
    transit_min: float = 10.0
    loiter_min: float = 2.0
    deploy_min: float = 0.5
    return_min: float = 2.0
    docking_assumed_min: float = 0.5
    redeploy_decision_min: float = 0.25
    return_home_min: float = 10.0
    max_duration_min: float = 180.0
    dt_min: float = 0.25
    max_deployment_cycles: int = 2

    def __post_init__(self) -> None:
        if self.max_deployment_cycles < 0:
            raise ValueError("max_deployment_cycles must be non-negative")
        for field_name in (
            "preflight_min",
            "transit_min",
            "loiter_min",
            "deploy_min",
            "return_min",
            "docking_assumed_min",
            "redeploy_decision_min",
            "return_home_min",
            "max_duration_min",
            "dt_min",
        ):
            if getattr(self, field_name) <= 0.0:
                raise ValueError(f"{field_name} must be positive")


@dataclass(frozen=True)
class MissionResult:
    time_min: np.ndarray
    mothership_energy_wh: np.ndarray
    quad_energy_wh: np.ndarray
    states: list[MissionState]
    completed_cycles: int
    aborted: bool
    warnings: list[str]
    solar_power_w: float
    mothership_cruise_rate_w: float
    mothership_charging_rate_w: float
    propulsion_power_w: float

    @property
    def final_mothership_battery_wh(self) -> float:
        return float(self.mothership_energy_wh[-1])

    @property
    def final_quad_battery_wh(self) -> float:
        return float(self.quad_energy_wh[-1])


def _append_sample(
    times: list[float],
    mothership_energy: list[float],
    quad_energy: list[float],
    states: list[MissionState],
    time_min: float,
    mothership_wh: float,
    quad_wh: float,
    state: MissionState,
) -> None:
    times.append(time_min)
    mothership_energy.append(mothership_wh)
    quad_energy.append(quad_wh)
    states.append(state)


def run_mission_energy_sim(
    quad: QuadcopterModel,
    mothership: MothershipEnergyParameters,
    solar: SolarParameters,
    timing: MissionTiming,
    propulsion_power_w: float | None = None,
) -> MissionResult:
    """Run a deterministic Phase 2 energy-loop simulation."""
    prop_power = propulsion_power_w or mothership.fallback_propulsion_power_w
    solar_w = solar_power_w(solar)
    cruise_rate_w = mothership_energy_rate_w(
        solar_w,
        prop_power,
        mothership.avionics_power_w,
    )
    charging_rate_w = mothership_energy_rate_w(
        solar_w,
        prop_power,
        mothership.avionics_power_w,
        mothership.charge_power_w,
    )

    times: list[float] = []
    mothership_energy: list[float] = []
    quad_energy: list[float] = []
    states: list[MissionState] = []
    warnings: list[str] = []

    time_min = 0.0
    mothership_wh = mothership.battery_capacity_wh
    quad_wh = quad.battery_usable_wh()
    completed_cycles = 0
    aborted = False

    def step_state(
        state: MissionState,
        duration_min: float,
        mothership_rate_w: float,
        quad_rate_w: float = 0.0,
        quad_charge_limit_wh: float | None = None,
    ) -> bool:
        nonlocal time_min, mothership_wh, quad_wh, aborted
        remaining = duration_min
        while remaining > 1e-9:
            dt = min(timing.dt_min, remaining)
            mothership_wh += mothership_rate_w * dt / 60.0
            quad_wh += quad_rate_w * dt / 60.0

            mothership_wh = min(mothership_wh, mothership.battery_capacity_wh)
            if quad_charge_limit_wh is None:
                quad_wh = min(quad_wh, quad.battery_usable_wh())
            else:
                quad_wh = min(quad_wh, quad_charge_limit_wh)

            time_min += dt
            if mothership_wh < 0.0 or quad_wh < 0.0:
                mothership_wh = max(0.0, mothership_wh)
                quad_wh = max(0.0, quad_wh)
                aborted = True
                _append_sample(
                    times,
                    mothership_energy,
                    quad_energy,
                    states,
                    time_min,
                    mothership_wh,
                    quad_wh,
                    MissionState.ABORT,
                )
                return False

            _append_sample(
                times,
                mothership_energy,
                quad_energy,
                states,
                time_min,
                mothership_wh,
                quad_wh,
                state,
            )
            remaining -= dt

            if time_min >= timing.max_duration_min:
                warnings.append("Mission reached max_duration_min before completion.")
                aborted = True
                return False
        return True

    initial_cruise_rate = cruise_rate_w
    if not step_state(MissionState.PREFLIGHT, timing.preflight_min, 0.0):
        pass
    elif not step_state(
        MissionState.MOTHERSHIP_TRANSIT, timing.transit_min, initial_cruise_rate
    ):
        pass
    elif not step_state(MissionState.LOITER, timing.loiter_min, cruise_rate_w):
        pass
    else:
        while completed_cycles < timing.max_deployment_cycles and not aborted:
            flight_rate = quad_flight_energy_rate_w(
                quad.flight_propulsion_power_w(),
                quad.params.payload_power_w,
            )
            if not step_state(
                MissionState.DEPLOY_QUAD,
                timing.deploy_min,
                cruise_rate_w,
                flight_rate,
            ):
                break
            if not step_state(
                MissionState.QUAD_INSPECT,
                quad.reserve_aware_inspection_min(),
                cruise_rate_w,
                flight_rate,
            ):
                break
            if not step_state(
                MissionState.QUAD_RETURN,
                timing.return_min,
                cruise_rate_w,
                flight_rate,
            ):
                break
            if not step_state(
                MissionState.DOCKING_ASSUMED,
                timing.docking_assumed_min,
                cruise_rate_w,
            ):
                break

            completed_cycles += 1

            charge_needed_wh = quad.battery_usable_wh() - quad_wh
            charge_rate_w = quad_charge_energy_rate_w(
                mothership.charge_power_w,
                mothership.charge_efficiency,
            )
            charge_duration_min = 0.0
            if charge_needed_wh > 1e-9:
                charge_duration_min = 60.0 * charge_needed_wh / charge_rate_w
            if not step_state(
                MissionState.CHARGING,
                charge_duration_min,
                charging_rate_w,
                charge_rate_w,
                quad.battery_usable_wh(),
            ):
                break
            if not step_state(
                MissionState.REDEPLOY_OR_RETURN,
                timing.redeploy_decision_min,
                cruise_rate_w,
            ):
                break

        if not aborted:
            if not step_state(MissionState.RETURN_HOME, timing.return_home_min, cruise_rate_w):
                pass
            else:
                step_state(MissionState.COMPLETE, timing.dt_min, 0.0)

    if completed_cycles < timing.max_deployment_cycles:
        warnings.append("Completed fewer deployment cycles than requested.")
    if mothership_wh < 0.15 * mothership.battery_capacity_wh:
        warnings.append("Final mothership battery is below 15% reserve.")
    if quad_wh < quad.params.reserve_fraction * quad.battery_usable_wh():
        warnings.append("Final quad battery is below configured reserve fraction.")
    if cruise_rate_w < 0.0:
        warnings.append("Solar is endurance assist only; cruise still drains battery.")

    return MissionResult(
        time_min=np.array(times),
        mothership_energy_wh=np.array(mothership_energy),
        quad_energy_wh=np.array(quad_energy),
        states=states,
        completed_cycles=completed_cycles,
        aborted=aborted,
        warnings=warnings,
        solar_power_w=solar_w,
        mothership_cruise_rate_w=cruise_rate_w,
        mothership_charging_rate_w=charging_rate_w,
        propulsion_power_w=prop_power,
    )
