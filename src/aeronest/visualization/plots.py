"""Plotting helpers for AeroNest simulation phases."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from aeronest.physics.fixed_wing import FixedWingModel
from aeronest.physics.quadcopter import QuadcopterModel
from aeronest.simulation.mission_sim import MissionResult


def save_phase_1_plots(model: FixedWingModel, sweep: dict, output_dir: Path) -> list[Path]:
    """Save Phase 1 fixed-wing performance plots and return their paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    speed = sweep["speed_m_s"]
    stall_speed = model.stall_speed_m_s()
    docking_speed = model.recommended_docking_speed_m_s()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(speed, sweep["lift_at_clmax_n"], label="Lift at C_Lmax")
    ax.axhline(model.weight_n, color="black", linestyle="--", label="Weight")
    ax.axvline(stall_speed, color="red", linestyle=":", label="Stall speed")
    ax.axvline(
        docking_speed,
        color="purple",
        linestyle="-.",
        label="Recommended docking speed",
    )
    ax.set_xlabel("Speed (m/s)")
    ax.set_ylabel("Force (N)")
    ax.set_title("Phase 1 Lift Margin")
    ax.grid(True, alpha=0.3)
    ax.legend()
    path = output_dir / "lift_margin.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(speed, sweep["drag_n"], label="Level-flight drag")
    ax.axvline(stall_speed, color="red", linestyle=":", label="Stall speed")
    ax.axvline(
        docking_speed,
        color="purple",
        linestyle="-.",
        label="Recommended docking speed",
    )
    ax.set_xlabel("Speed (m/s)")
    ax.set_ylabel("Drag (N)")
    ax.set_title("Phase 1 Drag Required")
    ax.grid(True, alpha=0.3)
    ax.legend()
    path = output_dir / "drag_required.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(speed, sweep["power_required_w"], label="Level-flight power")
    ax.axvline(stall_speed, color="red", linestyle=":", label="Stall speed")
    ax.axvline(
        docking_speed,
        color="purple",
        linestyle="-.",
        label="Recommended docking speed",
    )
    ax.set_xlabel("Speed (m/s)")
    ax.set_ylabel("Power required (W)")
    ax.set_title("Phase 1 Power Required")
    ax.grid(True, alpha=0.3)
    ax.legend()
    path = output_dir / "power_required.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    return paths


def save_phase_2_plots(
    quad: QuadcopterModel,
    result: MissionResult,
    output_dir: Path,
) -> list[Path]:
    """Save Phase 2 mission-energy plots and return their paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(result.time_min, result.quad_energy_wh, label="Quad battery")
    ax.axhline(quad.battery_usable_wh(), color="black", linestyle="--", label="Usable capacity")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Energy (Wh)")
    ax.set_title("Phase 2 Quadcopter Battery vs Time")
    ax.grid(True, alpha=0.3)
    ax.legend()
    path = output_dir / "quadcopter_battery_vs_time.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(result.time_min, result.mothership_energy_wh, label="Mothership battery")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Energy (Wh)")
    ax.set_title("Phase 2 Mothership Battery vs Time")
    ax.grid(True, alpha=0.3)
    ax.legend()
    path = output_dir / "mothership_battery_vs_time.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    state_names = list(dict.fromkeys(state.value for state in result.states))
    state_to_index = {name: index for index, name in enumerate(state_names)}
    state_indices = [state_to_index[state.value] for state in result.states]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.step(result.time_min, state_indices, where="post")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Mission state")
    ax.set_yticks(range(len(state_names)), state_names)
    ax.set_title("Phase 2 Mission State Timeline")
    ax.grid(True, axis="x", alpha=0.3)
    path = output_dir / "mission_state_timeline.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    masses = np.linspace(0.20, 0.50, 80)
    endurance = []
    for mass in masses:
        varied = QuadcopterModel(
            params=type(quad.params)(
                mass_kg=float(mass),
                n_rotors=quad.params.n_rotors,
                rotor_diameter_m=quad.params.rotor_diameter_m,
                battery_voltage_nominal_v=quad.params.battery_voltage_nominal_v,
                battery_capacity_ah=quad.params.battery_capacity_ah,
                usable_fraction=quad.params.usable_fraction,
                eta_motor_prop=quad.params.eta_motor_prop,
                avionics_power_w=quad.params.avionics_power_w,
                payload_power_w=quad.params.payload_power_w,
                maneuver_factor=quad.params.maneuver_factor,
                reserve_fraction=quad.params.reserve_fraction,
                gravity_m_s2=quad.params.gravity_m_s2,
            ),
            air_density_kg_m3=quad.air_density_kg_m3,
        )
        endurance.append(varied.endurance_min())
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(masses, endurance)
    ax.axvline(quad.params.mass_kg, color="black", linestyle="--", label="Configured mass")
    ax.set_xlabel("Quadcopter mass (kg)")
    ax.set_ylabel("Endurance (min)")
    ax.set_title("Phase 2 Endurance vs Mass")
    ax.grid(True, alpha=0.3)
    ax.legend()
    path = output_dir / "endurance_vs_mass.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    labels = [
        "Solar",
        "Mothership propulsion",
        "Mothership avionics",
        "Charge power",
        "Quad flight",
    ]
    values = [
        result.solar_power_w,
        -result.propulsion_power_w,
        -abs(result.mothership_cruise_rate_w - result.solar_power_w + result.propulsion_power_w),
        -abs(result.mothership_charging_rate_w - result.mothership_cruise_rate_w),
        -quad.flight_power_w(),
    ]
    colors = ["tab:green" if value >= 0.0 else "tab:red" for value in values]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(labels, values, color=colors)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel("Power (W)")
    ax.set_title("Phase 2 Energy Flow Summary")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, axis="y", alpha=0.3)
    path = output_dir / "energy_flow_summary.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    return paths
