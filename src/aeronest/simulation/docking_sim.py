"""2D docking kinematics simulation for Phase 3."""

from dataclasses import dataclass

import numpy as np

from aeronest.control.docking_controller import DockingController
from aeronest.perception.apriltag_model import AprilTagModel
from aeronest.perception.docking_pad_model import DockingPadModel
from aeronest.perception.uwb_model import UWBModel


@dataclass(frozen=True)
class DockingSimConfig:
    v_dock_m_s: float = 7.6
    initial_x_offset_m: float = -5.0
    initial_z_offset_m: float = 1.0
    dt_s: float = 0.02
    max_time_s: float = 20.0
    max_descent_rate_m_s: float = 0.35
    final_descent_rate_m_s: float = 0.20
    safe_x_error_m: float = 12.0
    safe_z_min_m: float = -0.35
    safe_z_max_m: float = 3.0
    abort_error_growth_m: float = 14.0
    final_stage_range_m: float = 0.55
    tag_lost_timeout_s: float = 0.75
    high_velocity_pad_range_m: float = 0.25
    wind_x_m_s2: float = 0.0
    wind_z_m_s2: float = 0.0
    gust_std_m_s2: float = 0.04
    sensor_noise_std_m: float = 0.03
    tag_dropout_probability: float = 0.03
    battery_reserve_ok: bool = True
    random_seed: int | None = 7


@dataclass(frozen=True)
class DockingResult:
    time_s: np.ndarray
    mothership_position_m: np.ndarray
    quad_position_m: np.ndarray
    relative_position_m: np.ndarray
    relative_velocity_m_s: np.ndarray
    tag_detection_probability: np.ndarray
    tag_detected: np.ndarray
    success: bool
    abort_reason: str
    time_to_dock_s: float

    @property
    def touchdown_x_error_m(self) -> float:
        return float(abs(self.relative_position_m[-1, 0]))

    @property
    def touchdown_z_error_m(self) -> float:
        return float(abs(self.relative_position_m[-1, 1]))

    @property
    def touchdown_relative_velocity_m_s(self) -> float:
        return float(np.linalg.norm(self.relative_velocity_m_s[-1]))


def _inside_capture_envelope(
    relative_position_m: np.ndarray,
    relative_velocity_m_s: np.ndarray,
    pad: DockingPadModel,
) -> bool:
    return (
        abs(relative_position_m[0]) < pad.x_tolerance_m
        and abs(relative_position_m[1]) < pad.z_tolerance_m
        and np.linalg.norm(relative_velocity_m_s) < pad.relative_velocity_tolerance_m_s
    )


def run_docking_sim(
    config: DockingSimConfig | None = None,
    controller: DockingController | None = None,
    tag_model: AprilTagModel | None = None,
    uwb_model: UWBModel | None = None,
    pad: DockingPadModel | None = None,
) -> DockingResult:
    """Run a 2D docking simulation with wind, sensor noise, and tag dropout."""
    config = config or DockingSimConfig()
    controller = controller or DockingController()
    tag_model = tag_model or AprilTagModel()
    uwb_model = uwb_model or UWBModel(noise_std_m=config.sensor_noise_std_m)
    pad = pad or DockingPadModel()
    rng = np.random.default_rng(config.random_seed)

    x_m = 0.0
    z_m = 0.0
    vx_m = config.v_dock_m_s
    vz_m = 0.0
    x_q = config.initial_x_offset_m
    z_q = config.initial_z_offset_m
    vx_q = config.v_dock_m_s
    vz_q = 0.0

    times: list[float] = []
    mothership_positions: list[list[float]] = []
    quad_positions: list[list[float]] = []
    rel_positions: list[list[float]] = []
    rel_velocities: list[list[float]] = []
    tag_probabilities: list[float] = []
    tag_detections: list[bool] = []

    stable_time_s = 0.0
    tag_lost_s = 0.0
    success = False
    abort_reason = "timeout"
    t = 0.0

    while t <= config.max_time_s:
        rel_pos = np.array([x_q - x_m, z_q - z_m], dtype=float)
        rel_vel = np.array([vx_q - vx_m, vz_q - vz_m], dtype=float)
        range_m = max(float(np.linalg.norm(rel_pos)), 1e-6)
        rel_speed = float(np.linalg.norm(rel_vel))
        tag_probability = tag_model.detection_probability(
            range_m,
            relative_velocity_m_s=rel_speed,
            dropout_probability=config.tag_dropout_probability,
        )
        tag_detected = bool(rng.random() < tag_probability)

        times.append(t)
        mothership_positions.append([x_m, z_m])
        quad_positions.append([x_q, z_q])
        rel_positions.append(rel_pos.tolist())
        rel_velocities.append(rel_vel.tolist())
        tag_probabilities.append(tag_probability)
        tag_detections.append(tag_detected)

        if not config.battery_reserve_ok:
            abort_reason = "battery_below_reserve"
            break
        if _inside_capture_envelope(rel_pos, rel_vel, pad):
            stable_time_s += config.dt_s
            if stable_time_s >= pad.stable_capture_time_s:
                success = True
                abort_reason = "success"
                break
        else:
            stable_time_s = 0.0

        if range_m < config.final_stage_range_m and not tag_detected:
            tag_lost_s += config.dt_s
            if tag_lost_s > config.tag_lost_timeout_s:
                abort_reason = "tag_lost_too_long"
                break
        else:
            tag_lost_s = 0.0

        if range_m < config.high_velocity_pad_range_m and rel_speed > 0.55:
            abort_reason = "relative_velocity_too_high_near_pad"
            break
        if abs(rel_pos[0]) > config.safe_x_error_m or not (
            config.safe_z_min_m <= rel_pos[1] <= config.safe_z_max_m
        ):
            abort_reason = "quad_exited_safe_envelope"
            break
        if range_m > config.abort_error_growth_m:
            abort_reason = "relative_error_grew_too_much"
            break
        if abs(config.wind_x_m_s2) + abs(config.wind_z_m_s2) > 2.5:
            abort_reason = "wind_disturbance_too_large"
            break

        if range_m < config.final_stage_range_m and tag_detected:
            measured_rel_pos = rel_pos + rng.normal(0.0, 0.01, size=2)
        else:
            measured_rel_pos = uwb_model.measure(rel_pos, rng)

        accel = controller.acceleration_command(measured_rel_pos, rel_vel)
        gust = rng.normal(0.0, config.gust_std_m_s2, size=2)
        disturbance = np.array([config.wind_x_m_s2, config.wind_z_m_s2]) + gust
        accel = accel + disturbance

        vx_q += accel[0] * config.dt_s
        vz_q += accel[1] * config.dt_s
        descent_limit = (
            config.final_descent_rate_m_s
            if range_m < config.final_stage_range_m
            else config.max_descent_rate_m_s
        )
        vz_q = max(vz_q, -descent_limit)

        x_m += vx_m * config.dt_s
        z_m += vz_m * config.dt_s
        x_q += vx_q * config.dt_s
        z_q += vz_q * config.dt_s
        t += config.dt_s

    return DockingResult(
        time_s=np.array(times),
        mothership_position_m=np.array(mothership_positions),
        quad_position_m=np.array(quad_positions),
        relative_position_m=np.array(rel_positions),
        relative_velocity_m_s=np.array(rel_velocities),
        tag_detection_probability=np.array(tag_probabilities),
        tag_detected=np.array(tag_detections, dtype=bool),
        success=success,
        abort_reason=abort_reason,
        time_to_dock_s=float(times[-1]) if times else 0.0,
    )
