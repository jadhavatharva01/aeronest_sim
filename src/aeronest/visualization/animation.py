"""Animation helpers for Phase 3 docking."""

from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
import numpy as np

from aeronest.simulation.docking_sim import DockingResult


def save_docking_animation(result: DockingResult, output_dir: Path) -> list[Path]:
    """Create GIF and optional MP4 docking animations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x position (m)")
    ax.set_ylabel("z position (m)")
    ax.set_title("Phase 3 Docking Animation")

    x_min = float(min(result.mothership_position_m[:, 0].min(), result.quad_position_m[:, 0].min())) - 1.0
    x_max = float(max(result.mothership_position_m[:, 0].max(), result.quad_position_m[:, 0].max())) + 1.0
    z_min = -0.5
    z_max = float(max(result.quad_position_m[:, 1].max(), 1.5)) + 0.5
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(z_min, z_max)
    ax.grid(True, alpha=0.25)

    mothership, = ax.plot([], [], "s", color="tab:blue", markersize=9, label="Mothership")
    pad, = ax.plot([], [], "-", color="black", linewidth=4, label="Docking pad")
    quad, = ax.plot([], [], "o", color="tab:orange", markersize=7, label="Quadcopter")
    trail, = ax.plot([], [], "-", color="tab:orange", alpha=0.45, linewidth=1)
    rel_line, = ax.plot([], [], "--", color="tab:gray", linewidth=1)
    text = ax.text(0.02, 0.95, "", transform=ax.transAxes, va="top")
    ax.legend(loc="lower right")

    frame_step = max(1, len(result.time_s) // 160)
    frame_indices = np.arange(0, len(result.time_s), frame_step)
    if frame_indices[-1] != len(result.time_s) - 1:
        frame_indices = np.append(frame_indices, len(result.time_s) - 1)

    def update(frame_index: int):
        m = result.mothership_position_m[frame_index]
        q = result.quad_position_m[frame_index]
        mothership.set_data([m[0]], [m[1]])
        pad.set_data([m[0] - 0.2, m[0] + 0.2], [m[1], m[1]])
        quad.set_data([q[0]], [q[1]])
        start = max(0, frame_index - 80)
        trail.set_data(
            result.quad_position_m[start : frame_index + 1, 0],
            result.quad_position_m[start : frame_index + 1, 1],
        )
        rel_line.set_data([m[0], q[0]], [m[1], q[1]])
        rel_error = np.linalg.norm(result.relative_position_m[frame_index])
        label = "SUCCESS" if result.success else f"FAIL: {result.abort_reason}"
        text.set_text(
            f"t = {result.time_s[frame_index]:.2f} s\n"
            f"relative error = {rel_error:.2f} m\n{label}"
        )
        return mothership, pad, quad, trail, rel_line, text

    animation = FuncAnimation(
        fig,
        update,
        frames=frame_indices,
        interval=40,
        blit=True,
    )

    gif_path = output_dir / "docking_animation.gif"
    animation.save(gif_path, writer=PillowWriter(fps=25))
    paths.append(gif_path)

    if shutil.which("ffmpeg"):
        mp4_path = output_dir / "docking_animation.mp4"
        animation.save(mp4_path, writer=FFMpegWriter(fps=25))
        paths.append(mp4_path)

    plt.close(fig)
    return paths
