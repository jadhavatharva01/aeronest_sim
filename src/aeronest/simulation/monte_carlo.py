"""Monte Carlo feasibility testing for Phase 3 docking."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from aeronest.perception.apriltag_model import AprilTagModel
from aeronest.simulation.docking_sim import DockingSimConfig, run_docking_sim


@dataclass(frozen=True)
class MonteCarloConfig:
    n_runs: int = 120
    random_seed: int = 21


def run_docking_monte_carlo(config: MonteCarloConfig | None = None) -> pd.DataFrame:
    """Run randomized docking simulations and return per-run metrics."""
    config = config or MonteCarloConfig()
    rng = np.random.default_rng(config.random_seed)
    rows: list[dict] = []

    for run_id in range(config.n_runs):
        wind_speed = float(rng.uniform(0.0, 0.9))
        wind_sign = float(rng.choice([-1.0, 1.0]))
        wind_angle = float(rng.uniform(-0.5, 0.5))
        sensor_noise = float(rng.uniform(0.01, 0.10))
        tag_dropout = float(rng.uniform(0.0, 0.18))
        initial_x = float(-5.0 + rng.normal(0.0, 0.75))
        initial_z = float(1.0 + rng.normal(0.0, 0.25))
        docking_speed = float(7.6 + rng.normal(0.0, 0.45))
        quad_mass = float(0.30 + rng.normal(0.0, 0.035))
        tag_size = float(rng.choice([0.020, 0.025, 0.030]))

        mass_penalty = max(0.0, quad_mass - 0.30) * 2.5
        sim_config = DockingSimConfig(
            v_dock_m_s=docking_speed,
            initial_x_offset_m=initial_x,
            initial_z_offset_m=max(0.35, initial_z),
            wind_x_m_s2=wind_sign * wind_speed * (1.0 + wind_angle),
            wind_z_m_s2=0.25 * wind_speed * wind_angle - mass_penalty,
            gust_std_m_s2=float(rng.uniform(0.02, 0.12)),
            sensor_noise_std_m=sensor_noise,
            tag_dropout_probability=tag_dropout,
            random_seed=int(rng.integers(0, 2**31 - 1)),
        )
        result = run_docking_sim(
            config=sim_config,
            tag_model=AprilTagModel(tag_size_m=tag_size),
        )
        rows.append(
            {
                "run_id": run_id,
                "success": result.success,
                "touchdown_x_error_m": result.touchdown_x_error_m,
                "touchdown_z_error_m": result.touchdown_z_error_m,
                "touchdown_relative_velocity_m_s": (
                    result.touchdown_relative_velocity_m_s
                ),
                "time_to_dock_s": result.time_to_dock_s,
                "abort_reason": result.abort_reason,
                "wind_speed_m_s2": wind_speed,
                "wind_direction_sign": wind_sign,
                "sensor_noise_std_m": sensor_noise,
                "tag_dropout_probability": tag_dropout,
                "initial_x_offset_m": initial_x,
                "initial_z_offset_m": initial_z,
                "docking_speed_m_s": docking_speed,
                "quad_mass_kg": quad_mass,
                "tag_size_m": tag_size,
            }
        )

    return pd.DataFrame(rows)


def success_rate(data: pd.DataFrame) -> float:
    """Return scalar success rate from Monte Carlo results."""
    if data.empty:
        return 0.0
    return float(data["success"].mean())
