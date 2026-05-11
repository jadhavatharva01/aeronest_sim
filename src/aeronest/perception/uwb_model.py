"""UWB/telemetry placeholder for non-final-stage relative localization."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class UWBModel:
    """Gaussian relative-position measurement placeholder."""

    noise_std_m: float = 0.12

    def measure(self, relative_position_m: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if self.noise_std_m <= 0.0:
            return np.asarray(relative_position_m, dtype=float)
        return np.asarray(relative_position_m, dtype=float) + rng.normal(
            0.0,
            self.noise_std_m,
            size=2,
        )
