"""Finger-sized AprilTag detection model for final docking alignment."""

from dataclasses import dataclass
from math import cos, radians


@dataclass(frozen=True)
class AprilTagModel:
    """Simple range, blur, angle, and lighting detection model."""

    tag_size_m: float = 0.025
    f_px: float = 500.0
    lighting_factor: float = 0.85

    def __post_init__(self) -> None:
        if self.tag_size_m <= 0.0:
            raise ValueError("tag_size_m must be positive")
        if self.f_px <= 0.0:
            raise ValueError("f_px must be positive")
        if not 0.0 <= self.lighting_factor <= 1.0:
            raise ValueError("lighting_factor must be between 0 and 1")

    def tag_pixel_size(self, range_m: float) -> float:
        """Return apparent tag size in pixels."""
        if range_m <= 0.0:
            raise ValueError("range_m must be positive")
        return self.f_px * self.tag_size_m / range_m

    def detection_probability(
        self,
        range_m: float,
        relative_velocity_m_s: float = 0.0,
        viewing_angle_deg: float = 0.0,
        lighting_factor: float | None = None,
        dropout_probability: float = 0.0,
    ) -> float:
        """Return probability of detecting the tag in one frame."""
        pixels = self.tag_pixel_size(range_m)
        if pixels > 60.0:
            base = 0.95
        elif pixels >= 40.0:
            base = 0.70
        elif pixels >= 25.0:
            base = 0.35
        else:
            base = 0.0

        lighting = self.lighting_factor if lighting_factor is None else lighting_factor
        lighting = min(max(lighting, 0.0), 1.0)
        dropout = min(max(dropout_probability, 0.0), 1.0)
        blur_penalty = max(0.0, 1.0 - 0.55 * max(relative_velocity_m_s, 0.0))
        angle_penalty = max(0.0, cos(radians(abs(viewing_angle_deg)))) ** 1.5
        probability = base * blur_penalty * angle_penalty * lighting * (1.0 - dropout)
        return min(max(probability, 0.0), 1.0)
