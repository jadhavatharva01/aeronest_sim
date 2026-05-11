from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aeronest.perception.apriltag_model import AprilTagModel


def test_tag_pixel_size_decreases_with_range() -> None:
    model = AprilTagModel()
    assert model.tag_pixel_size(0.25) > model.tag_pixel_size(0.50)
    assert model.tag_pixel_size(0.50) > model.tag_pixel_size(1.0)


def test_detection_probability_decreases_with_range() -> None:
    model = AprilTagModel(lighting_factor=1.0)
    near = model.detection_probability(0.20)
    mid = model.detection_probability(0.35)
    far = model.detection_probability(0.75)
    assert near > mid > far
    assert far == 0.0


def test_motion_blur_penalizes_detection_probability() -> None:
    model = AprilTagModel(lighting_factor=1.0)
    slow = model.detection_probability(0.25, relative_velocity_m_s=0.1)
    fast = model.detection_probability(0.25, relative_velocity_m_s=1.0)
    assert slow > fast
