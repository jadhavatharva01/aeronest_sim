"""Atmosphere utilities for early AeroNest physics phases."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Atmosphere:
    """Simple constant-density atmosphere model."""

    rho: float = 1.20

    def dynamic_pressure(self, speed_m_s: float) -> float:
        """Return dynamic pressure in pascals."""
        if speed_m_s < 0.0:
            raise ValueError("speed_m_s must be non-negative")
        return 0.5 * self.rho * speed_m_s**2
