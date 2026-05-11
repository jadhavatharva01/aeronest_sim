"""PD docking controller for 2D relative motion."""

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class DockingController:
    """Acceleration-limited PD controller in the relative x-z frame."""

    kp: np.ndarray = field(default_factory=lambda: np.array([0.65, 0.75]))
    kd: np.ndarray = field(default_factory=lambda: np.array([1.55, 1.45]))
    max_accel_m_s2: float = 3.0

    def acceleration_command(
        self,
        relative_position_m: np.ndarray,
        relative_velocity_m_s: np.ndarray,
        feedforward_m_s2: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return acceleration command that drives relative state toward zero."""
        feedforward = (
            np.zeros(2, dtype=float)
            if feedforward_m_s2 is None
            else np.asarray(feedforward_m_s2, dtype=float)
        )
        position_error = -np.asarray(relative_position_m, dtype=float)
        velocity_error = -np.asarray(relative_velocity_m_s, dtype=float)
        accel = self.kp * position_error + self.kd * velocity_error + feedforward
        norm = float(np.linalg.norm(accel))
        if norm > self.max_accel_m_s2:
            accel = accel * self.max_accel_m_s2 / norm
        return accel
