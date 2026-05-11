from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aeronest.perception.apriltag_model import AprilTagModel
from aeronest.simulation.docking_sim import DockingSimConfig, run_docking_sim
from aeronest.simulation.monte_carlo import MonteCarloConfig, run_docking_monte_carlo
from aeronest.visualization.animation import save_docking_animation


def test_monte_carlo_returns_expected_columns() -> None:
    data = run_docking_monte_carlo(MonteCarloConfig(n_runs=5, random_seed=3))
    expected_columns = {
        "run_id",
        "success",
        "touchdown_x_error_m",
        "touchdown_z_error_m",
        "touchdown_relative_velocity_m_s",
        "time_to_dock_s",
        "abort_reason",
        "wind_speed_m_s2",
        "wind_direction_sign",
        "sensor_noise_std_m",
        "tag_dropout_probability",
        "initial_x_offset_m",
        "initial_z_offset_m",
        "docking_speed_m_s",
        "quad_mass_kg",
        "tag_size_m",
    }
    assert expected_columns.issubset(set(data.columns))
    assert len(data) == 5


def test_animation_function_creates_file(tmp_path: Path) -> None:
    result = run_docking_sim(
        DockingSimConfig(
            max_time_s=8.0,
            gust_std_m_s2=0.0,
            sensor_noise_std_m=0.0,
            tag_dropout_probability=0.0,
            random_seed=4,
        ),
        tag_model=AprilTagModel(),
    )
    paths = save_docking_animation(result, tmp_path)
    gif_paths = [path for path in paths if path.suffix == ".gif"]
    assert gif_paths
    assert gif_paths[0].exists()
    assert gif_paths[0].stat().st_size > 0
