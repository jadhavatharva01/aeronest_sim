#!/usr/bin/env python
"""Run Phase 2 quadcopter and mission energy simulation."""

from pathlib import Path
import sys

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aeronest.physics.atmosphere import Atmosphere
from aeronest.physics.energy import MothershipEnergyParameters, SolarParameters
from aeronest.physics.fixed_wing import FixedWingModel, FixedWingParameters
from aeronest.physics.quadcopter import QuadcopterModel, QuadcopterParameters
from aeronest.simulation.mission_sim import MissionTiming, run_mission_energy_sim
from aeronest.visualization.plots import save_phase_2_plots


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_models() -> tuple[
    QuadcopterModel,
    MothershipEnergyParameters,
    SolarParameters,
    MissionTiming,
    float,
]:
    vehicle_config = load_yaml(REPO_ROOT / "config" / "vehicle_params.yaml")
    mission_config = load_yaml(REPO_ROOT / "config" / "mission_params.yaml")
    sim_config = load_yaml(REPO_ROOT / "config" / "sim_params.yaml")

    quad_cfg = vehicle_config["quadcopter"]
    quad = QuadcopterModel(
        params=QuadcopterParameters(**quad_cfg),
        air_density_kg_m3=sim_config["simulation"]["air_density_kg_m3"],
    )

    mothership = MothershipEnergyParameters(**vehicle_config["mothership_energy"])
    solar = SolarParameters(**vehicle_config["solar"])
    timing = MissionTiming(**mission_config["phase_2"])

    fixed_wing_cfg = vehicle_config["fixed_wing_mothership"]
    fixed_wing = FixedWingModel(
        params=FixedWingParameters(**fixed_wing_cfg),
        atmosphere=Atmosphere(rho=sim_config["simulation"]["air_density_kg_m3"]),
    )
    cruise_speed = fixed_wing.recommended_docking_speed_m_s(
        mission_config["mission"]["docking_speed_factor"]
    )
    propulsion_power_w = fixed_wing.power_required_w(cruise_speed)

    return quad, mothership, solar, timing, propulsion_power_w


def main() -> None:
    quad, mothership, solar, timing, propulsion_power_w = build_models()
    result = run_mission_energy_sim(
        quad=quad,
        mothership=mothership,
        solar=solar,
        timing=timing,
        propulsion_power_w=propulsion_power_w,
    )

    print(f"Quad hover power: {quad.electrical_hover_power_w():.2f} W")
    print(f"Quad estimated flight power: {quad.flight_power_w():.2f} W")
    print(f"Quad total energy: {quad.battery_total_wh():.2f} Wh")
    print(f"Quad usable energy: {quad.battery_usable_wh():.2f} Wh")
    print(
        "Reserve-aware inspection time: "
        f"{quad.reserve_aware_inspection_min():.2f} min"
    )
    print(f"Mothership solar power: {result.solar_power_w:.2f} W")
    print(
        "Mothership net power during cruise: "
        f"{result.mothership_cruise_rate_w:.2f} W"
    )
    print(
        "Mothership net power during charging: "
        f"{result.mothership_charging_rate_w:.2f} W"
    )
    print(f"Number of completed deployment cycles: {result.completed_cycles}")
    print(f"Final mothership battery: {result.final_mothership_battery_wh:.2f} Wh")
    print(f"Final quad battery: {result.final_quad_battery_wh:.2f} Wh")

    warnings = list(result.warnings)
    if result.aborted:
        warnings.append("Mission aborted due to energy depletion or duration limit.")
    if result.completed_cycles < timing.max_deployment_cycles:
        warnings.append("Energy infeasible for requested deployment cycles.")

    if warnings:
        print("Warnings:")
        for warning in dict.fromkeys(warnings):
            print(f"- {warning}")

    plot_paths = save_phase_2_plots(quad, result, REPO_ROOT / "outputs" / "phase_2")
    print("Saved plots:")
    for path in plot_paths:
        print(f"- {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
