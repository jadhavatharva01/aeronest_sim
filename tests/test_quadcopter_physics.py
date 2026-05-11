import math
from pathlib import Path
import sys

import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aeronest.physics.quadcopter import QuadcopterModel, QuadcopterParameters


@pytest.fixture
def quad() -> QuadcopterModel:
    return QuadcopterModel(QuadcopterParameters(), air_density_kg_m3=1.20)


def test_rotor_disk_area_matches_total_rotor_area(quad: QuadcopterModel) -> None:
    expected = 4 * math.pi * (0.12 / 2.0) ** 2
    assert quad.rotor_disk_area_m2 == pytest.approx(expected)


def test_hover_power_sanity_for_default_quad(quad: QuadcopterModel) -> None:
    expected_ideal = quad.weight_n**1.5 / math.sqrt(
        2.0 * quad.air_density_kg_m3 * quad.rotor_disk_area_m2
    )
    expected_electrical = (
        expected_ideal / quad.params.eta_motor_prop + quad.params.avionics_power_w
    )
    assert quad.ideal_hover_power_w() > 0.0
    assert quad.electrical_hover_power_w() > quad.ideal_hover_power_w()
    assert quad.ideal_hover_power_w() == pytest.approx(expected_ideal)
    assert quad.electrical_hover_power_w() == pytest.approx(expected_electrical)


def test_battery_wh_calculation(quad: QuadcopterModel) -> None:
    assert quad.battery_total_wh() == pytest.approx(7.38 * 2.45)
    assert quad.battery_usable_wh() == pytest.approx(7.38 * 2.45 * 0.80)


def test_reserve_aware_inspection_time_is_less_than_total_endurance(
    quad: QuadcopterModel,
) -> None:
    assert quad.reserve_aware_inspection_min() < quad.endurance_min()
    assert quad.reserve_aware_inspection_min() == pytest.approx(
        quad.endurance_min() * (1.0 - quad.params.reserve_fraction)
    )
