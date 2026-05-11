"""Plotting helpers for AeroNest simulation phases."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aeronest.physics.fixed_wing import FixedWingModel
from aeronest.physics.quadcopter import QuadcopterModel
from aeronest.perception.apriltag_model import AprilTagModel
from aeronest.simulation.docking_sim import DockingResult
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


def save_phase_3_docking_plots(
    result: DockingResult,
    tag_model: AprilTagModel,
    output_dir: Path,
) -> list[Path]:
    """Save single-run Phase 3 docking plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        result.mothership_position_m[:, 0],
        result.mothership_position_m[:, 1],
        label="Mothership",
    )
    ax.plot(result.quad_position_m[:, 0], result.quad_position_m[:, 1], label="Quad")
    ax.scatter(
        result.quad_position_m[0, 0],
        result.quad_position_m[0, 1],
        marker="o",
        color="tab:green",
        label="Quad start",
        zorder=3,
    )
    ax.scatter(
        result.quad_position_m[-1, 0],
        result.quad_position_m[-1, 1],
        marker="x",
        color="tab:red",
        label="Touchdown/final",
        zorder=3,
    )
    ax.set_xlabel("x position (m)")
    ax.set_ylabel("z position (m)")
    ax.set_title("Phase 3 Docking Trajectory")
    ax.grid(True, alpha=0.3)
    ax.legend()
    path = output_dir / "docking_trajectory.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    error_norm = np.linalg.norm(result.relative_position_m, axis=1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(result.time_s, result.relative_position_m[:, 0], label="x error")
    ax.plot(result.time_s, result.relative_position_m[:, 1], label="z error")
    ax.plot(result.time_s, error_norm, label="error norm")
    ax.axhline(0.10, color="tab:blue", linestyle=":", label="x success threshold")
    ax.axhline(-0.10, color="tab:blue", linestyle=":")
    ax.axhline(0.05, color="tab:orange", linestyle=":", label="z success threshold")
    ax.axhline(-0.05, color="tab:orange", linestyle=":")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Relative position (m)")
    ax.set_title("Phase 3 Relative Error vs Time")
    ax.grid(True, alpha=0.3)
    ax.legend()
    path = output_dir / "relative_error_vs_time.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    velocity_norm = np.linalg.norm(result.relative_velocity_m_s, axis=1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(result.time_s, result.relative_velocity_m_s[:, 0], label="x relative velocity")
    ax.plot(result.time_s, result.relative_velocity_m_s[:, 1], label="z relative velocity")
    ax.plot(result.time_s, velocity_norm, label="relative speed")
    ax.axhline(0.25, color="black", linestyle="--", label="capture limit")
    below_capture = np.flatnonzero(velocity_norm < 0.25)
    if len(below_capture) > 0:
        first_index = int(below_capture[0])
        ax.axvline(
            result.time_s[first_index],
            color="tab:green",
            linestyle=":",
            label="first below capture limit",
        )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Relative velocity (m/s)")
    ax.set_title("Phase 3 Relative Velocity vs Time")
    ax.grid(True, alpha=0.3)
    ax.legend()
    path = output_dir / "relative_velocity_vs_time.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    ranges = np.linspace(0.15, 1.2, 150)
    probabilities = [tag_model.detection_probability(float(r)) for r in ranges]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ranges, probabilities)
    ax.axvspan(0.2, 0.5, color="tab:green", alpha=0.12, label="rough reliable range")
    ax.set_xlabel("Range (m)")
    ax.set_ylabel("Detection probability")
    ax.set_title("Phase 3 AprilTag Detection Probability")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()
    path = output_dir / "apriltag_detection_probability.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    return paths


def save_phase_3_monte_carlo_plots(data: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Save Phase 3 Monte Carlo summary plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    wind_bins = pd.cut(data["wind_speed_m_s2"], bins=6)
    wind_success = data.groupby(wind_bins, observed=True)["success"].mean()
    wind_centers = [interval.mid for interval in wind_success.index]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(wind_centers, wind_success.values, marker="o")
    ax.set_xlabel("Wind disturbance magnitude (m/s^2)")
    ax.set_ylabel("Success rate")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Phase 3 Success Rate vs Wind")
    ax.grid(True, alpha=0.3)
    path = output_dir / "success_rate_vs_wind.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    speed_bins = pd.cut(data["docking_speed_m_s"], bins=6)
    speed_success = data.groupby(speed_bins, observed=True)["success"].mean()
    speed_centers = [interval.mid for interval in speed_success.index]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(speed_centers, speed_success.values, marker="o")
    ax.set_xlabel("Docking speed (m/s)")
    ax.set_ylabel("Success rate")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Phase 3 Success Rate vs Docking Speed")
    ax.grid(True, alpha=0.3)
    path = output_dir / "success_rate_vs_docking_speed.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(8, 5))
    successful = data[data["success"]]
    failed = data[~data["success"]]
    ax.scatter(
        successful["touchdown_x_error_m"],
        successful["touchdown_z_error_m"],
        alpha=0.7,
        label="Success",
    )
    if not failed.empty:
        ax.scatter(
            failed["touchdown_x_error_m"],
            failed["touchdown_z_error_m"],
            alpha=0.45,
            label="Failure",
        )
    ax.set_xlabel("Touchdown x error (m)")
    ax.set_ylabel("Touchdown z error (m)")
    ax.set_title("Phase 3 Touchdown Error Distribution")
    ax.grid(True, alpha=0.3)
    ax.legend()
    path = output_dir / "touchdown_error_distribution.png"
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
