#!/usr/bin/env python
"""Run Phase 3 Monte Carlo docking feasibility tests."""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aeronest.simulation.monte_carlo import MonteCarloConfig, run_docking_monte_carlo
from aeronest.visualization.plots import save_phase_3_monte_carlo_plots


def main() -> None:
    output_dir = REPO_ROOT / "outputs" / "phase_3"
    data = run_docking_monte_carlo(MonteCarloConfig(n_runs=120, random_seed=42))
    success_rate = data["success"].mean()

    print(f"Monte Carlo runs: {len(data)}")
    print(f"Success rate: {100.0 * success_rate:.1f}%")
    print(
        "Median touchdown x error: "
        f"{data['touchdown_x_error_m'].median():.3f} m"
    )
    print(
        "Median touchdown z error: "
        f"{data['touchdown_z_error_m'].median():.3f} m"
    )
    print(
        "Median touchdown relative velocity: "
        f"{data['touchdown_relative_velocity_m_s'].median():.3f} m/s"
    )
    print(f"Median time to dock: {data['time_to_dock_s'].median():.2f} s")
    print("Abort reasons:")
    for reason, count in data["abort_reason"].value_counts().items():
        print(f"- {reason}: {count}")
    print("Success rate by tag size:")
    for tag_size, rate in data.groupby("tag_size_m")["success"].mean().items():
        print(f"- {tag_size:.3f} m: {100.0 * rate:.1f}%")

    plot_paths = save_phase_3_monte_carlo_plots(data, output_dir)
    print("Saved plots:")
    for path in plot_paths:
        print(f"- {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
