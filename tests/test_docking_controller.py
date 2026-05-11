from pathlib import Path
import sys

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aeronest.control.docking_controller import DockingController
from aeronest.simulation.docking_sim import DockingSimConfig, run_docking_sim


def test_pd_controller_reduces_relative_error_in_no_wind_case() -> None:
    result = run_docking_sim(
        DockingSimConfig(
            gust_std_m_s2=0.0,
            sensor_noise_std_m=0.0,
            tag_dropout_probability=0.0,
            random_seed=1,
        )
    )

    initial_error = np.linalg.norm(result.relative_position_m[0])
    final_error = np.linalg.norm(result.relative_position_m[-1])
    assert final_error < initial_error
    assert result.success


def test_controller_respects_acceleration_limit() -> None:
    controller = DockingController(max_accel_m_s2=3.0)
    accel = controller.acceleration_command(
        relative_position_m=np.array([-20.0, 5.0]),
        relative_velocity_m_s=np.array([-3.0, 1.0]),
    )
    assert np.linalg.norm(accel) <= 3.0 + 1e-12


def test_docking_success_condition() -> None:
    result = run_docking_sim(
        DockingSimConfig(
            gust_std_m_s2=0.0,
            sensor_noise_std_m=0.0,
            tag_dropout_probability=0.0,
            random_seed=2,
        )
    )

    assert result.success
    assert result.abort_reason == "success"
    assert result.touchdown_x_error_m < 0.10
    assert result.touchdown_z_error_m < 0.05
    assert result.touchdown_relative_velocity_m_s < 0.25
