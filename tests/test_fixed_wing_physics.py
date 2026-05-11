import math
from pathlib import Path
import sys

import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aeronest.physics.atmosphere import Atmosphere
from aeronest.physics.fixed_wing import FixedWingModel, FixedWingParameters


@pytest.fixture
def model() -> FixedWingModel:
    return FixedWingModel(
        params=FixedWingParameters(),
        atmosphere=Atmosphere(rho=1.20),
    )


def test_effective_wing_area_includes_stacked_interference(model: FixedWingModel) -> None:
    assert model.effective_wing_area_m2 == pytest.approx(3 * 1.5 * 0.18 * 0.75)


def test_stall_speed_uses_weight_density_area_and_clmax(model: FixedWingModel) -> None:
    expected = math.sqrt(
        (2.0 * model.weight_n)
        / (1.20 * model.effective_wing_area_m2 * model.params.cl_max)
    )
    assert model.stall_speed_m_s() == pytest.approx(expected)


def test_recommended_docking_speed_is_25_percent_above_stall(
    model: FixedWingModel,
) -> None:
    assert model.recommended_docking_speed_m_s() == pytest.approx(
        1.25 * model.stall_speed_m_s()
    )


def test_lift_matches_dynamic_pressure_area_and_cl(model: FixedWingModel) -> None:
    speed = 7.5
    cl = 1.1
    expected = 0.5 * 1.20 * speed**2 * model.effective_wing_area_m2 * cl
    assert model.lift_n(speed, cl) == pytest.approx(expected)


def test_drag_and_power_are_positive_for_level_flight(model: FixedWingModel) -> None:
    speed = model.recommended_docking_speed_m_s()
    cl = model.lift_coefficient_for_level_flight(speed)
    assert model.drag_n(speed, cl) > 0.0
    assert model.power_required_w(speed) > 0.0


def test_drag_coefficient_matches_parasite_plus_induced_drag(
    model: FixedWingModel,
) -> None:
    cl = 1.05
    expected = model.params.cd0 + cl**2 / (
        math.pi * model.aspect_ratio * model.params.oswald_efficiency
    )
    assert model.drag_coefficient(cl) == pytest.approx(expected)


def test_power_required_matches_drag_times_speed_over_prop_efficiency(
    model: FixedWingModel,
) -> None:
    speed = 8.0
    cl = model.lift_coefficient_for_level_flight(speed)
    drag = model.drag_n(speed, cl)
    expected = drag * speed / model.params.propeller_efficiency
    assert model.power_required_w(speed) == pytest.approx(expected)


def test_level_flight_cl_decreases_with_speed(model: FixedWingModel) -> None:
    low_speed_cl = model.lift_coefficient_for_level_flight(7.0)
    high_speed_cl = model.lift_coefficient_for_level_flight(10.0)
    assert high_speed_cl < low_speed_cl


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    (
        ("interference_factor", 0.0),
        ("interference_factor", 1.1),
        ("oswald_efficiency", 0.0),
        ("oswald_efficiency", 1.1),
        ("propeller_efficiency", 0.0),
        ("propeller_efficiency", 1.1),
    ),
)
def test_efficiency_like_parameters_must_be_in_unit_interval(
    field_name: str,
    invalid_value: float,
) -> None:
    kwargs = {field_name: invalid_value}
    with pytest.raises(ValueError, match="greater than 0 and at most 1"):
        FixedWingParameters(**kwargs)
