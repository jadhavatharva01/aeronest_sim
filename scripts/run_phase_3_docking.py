#!/usr/bin/env python
"""Run a single Phase 3 docking simulation and animation."""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aeronest.perception.apriltag_model import AprilTagModel
from aeronest.simulation.docking_sim import DockingSimConfig, run_docking_sim
from aeronest.visualization.animation import save_docking_animation
from aeronest.visualization.plots import save_phase_3_docking_plots


def main() -> None:
    output_dir = REPO_ROOT / "outputs" / "phase_3"
    tag_model = AprilTagModel()
    result = run_docking_sim(
        DockingSimConfig(
            v_dock_m_s=7.6,
            initial_x_offset_m=-5.0,
            initial_z_offset_m=1.0,
            random_seed=5,
            tag_dropout_probability=0.01,
            gust_std_m_s2=0.02,
        ),
        tag_model=tag_model,
    )

    print(f"Docking success: {result.success}")
    print(f"Abort reason: {result.abort_reason}")
    print(f"Time to dock: {result.time_to_dock_s:.2f} s")
    print(f"Touchdown x error: {result.touchdown_x_error_m:.3f} m")
    print(f"Touchdown z error: {result.touchdown_z_error_m:.3f} m")
    print(
        "Touchdown relative velocity: "
        f"{result.touchdown_relative_velocity_m_s:.3f} m/s"
    )

    plot_paths = save_phase_3_docking_plots(result, tag_model, output_dir)
    animation_paths = save_docking_animation(result, output_dir)

    print("Saved plots:")
    for path in plot_paths:
        print(f"- {path.relative_to(REPO_ROOT)}")
    print("Saved animations:")
    for path in animation_paths:
        print(f"- {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
