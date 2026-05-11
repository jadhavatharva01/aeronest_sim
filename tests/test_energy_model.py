from pathlib import Path
import sys

import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aeronest.physics.energy import (
    SolarParameters,
    mothership_energy_rate_w,
    quad_charge_energy_rate_w,
    quad_flight_energy_rate_w,
    solar_power_w,
)


def test_solar_power_calculation() -> None:
    params = SolarParameters()
    expected = 0.22 * 0.35 * 800.0 * 0.75 * 0.75 * 0.92
    assert solar_power_w(params) == pytest.approx(expected)


def test_mothership_energy_rate_includes_solar_loads_and_charge() -> None:
    rate = mothership_energy_rate_w(
        solar_power=30.0,
        propulsion_power_w=40.0,
        avionics_power_w=8.0,
        charge_power_w=15.0,
    )
    assert rate == pytest.approx(30.0 - 40.0 - 8.0 - 15.0)


def test_quad_flight_and_charge_energy_rates() -> None:
    assert quad_flight_energy_rate_w(44.0, 2.0) == pytest.approx(-46.0)
    assert quad_charge_energy_rate_w(15.0, 0.85) == pytest.approx(12.75)
