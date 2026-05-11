"""Plotting helpers for AeroNest simulation phases."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from aeronest.physics.fixed_wing import FixedWingModel


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
