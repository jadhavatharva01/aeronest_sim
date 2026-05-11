#!/usr/bin/env python
"""Run Phase 1 fixed-wing slow-flight analysis."""

from pathlib import Path
import sys

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aeronest.physics.atmosphere import Atmosphere
from aeronest.physics.fixed_wing import FixedWingModel, FixedWingParameters
from aeronest.visualization.plots import save_phase_1_plots


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_model() -> tuple[FixedWingModel, dict, dict]:
    vehicle_config = load_yaml(REPO_ROOT / "config" / "vehicle_params.yaml")
    mission_config = load_yaml(REPO_ROOT / "config" / "mission_params.yaml")
    sim_config = load_yaml(REPO_ROOT / "config" / "sim_params.yaml")

    vehicle = vehicle_config["fixed_wing_mothership"]
    params = FixedWingParameters(
        n_wings=vehicle["n_wings"],
        span_m=vehicle["span_m"],
        chord_m=vehicle["chord_m"],
        interference_factor=vehicle["interference_factor"],
        mass_kg=vehicle["mass_kg"],
        cl_max=vehicle["cl_max"],
        cd0=vehicle["cd0"],
        oswald_efficiency=vehicle["oswald_efficiency"],
        propeller_efficiency=vehicle["propeller_efficiency"],
        gravity_m_s2=vehicle["gravity_m_s2"],
    )
    atmosphere = Atmosphere(rho=sim_config["simulation"]["air_density_kg_m3"])
    return FixedWingModel(params=params, atmosphere=atmosphere), mission_config, sim_config


def main() -> None:
    model, mission_config, sim_config = build_model()

    mission = mission_config["mission"]
    stall_speed = model.stall_speed_m_s()
    docking_speed = model.recommended_docking_speed_m_s(
        mission["docking_speed_factor"]
    )

    print(f"Stall speed: {stall_speed:.2f} m/s")
    print(f"Recommended docking speed: {docking_speed:.2f} m/s")
    if docking_speed > mission["docking_speed_warning_m_s"]:
        print(
            "WARNING: recommended docking speed exceeds "
            f"{mission['docking_speed_warning_m_s']:.1f} m/s"
        )

    speed_config = sim_config["simulation"]["speed_sweep_m_s"]
    speeds = np.linspace(
        speed_config["min"],
        speed_config["max"],
        speed_config["count"],
    )
    sweep = model.performance_sweep(speeds)

    output_dir = REPO_ROOT / sim_config["simulation"]["output_dir"]
    plot_paths = save_phase_1_plots(model, sweep, output_dir)
    print("Saved plots:")
    for path in plot_paths:
        print(f"- {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
