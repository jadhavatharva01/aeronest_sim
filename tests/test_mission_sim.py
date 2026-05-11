from pathlib import Path
import sys

import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aeronest.control.state_machine import MissionState
from aeronest.physics.energy import MothershipEnergyParameters, SolarParameters
from aeronest.physics.quadcopter import QuadcopterModel, QuadcopterParameters
from aeronest.simulation.mission_sim import MissionTiming, run_mission_energy_sim


def test_mission_state_progression() -> None:
    result = run_mission_energy_sim(
        quad=QuadcopterModel(QuadcopterParameters(), air_density_kg_m3=1.20),
        mothership=MothershipEnergyParameters(),
        solar=SolarParameters(),
        timing=MissionTiming(max_deployment_cycles=1),
        propulsion_power_w=44.0,
    )

    assert result.states[0] == MissionState.PREFLIGHT
    assert MissionState.MOTHERSHIP_TRANSIT in result.states
    assert MissionState.QUAD_INSPECT in result.states
    assert MissionState.DOCKING_ASSUMED in result.states
    assert MissionState.CHARGING in result.states
    assert result.states[-1] == MissionState.COMPLETE
    assert result.completed_cycles == 1
    assert not result.aborted


def test_energy_never_exceeds_battery_capacity() -> None:
    quad = QuadcopterModel(QuadcopterParameters(), air_density_kg_m3=1.20)
    mothership = MothershipEnergyParameters()
    result = run_mission_energy_sim(
        quad=quad,
        mothership=mothership,
        solar=SolarParameters(),
        timing=MissionTiming(max_deployment_cycles=1),
        propulsion_power_w=44.0,
    )

    assert result.mothership_energy_wh.max() <= mothership.battery_capacity_wh + 1e-9
    assert result.quad_energy_wh.max() <= quad.battery_usable_wh() + 1e-9


def test_energy_never_drops_below_zero_unless_abort_is_triggered() -> None:
    result = run_mission_energy_sim(
        quad=QuadcopterModel(QuadcopterParameters(), air_density_kg_m3=1.20),
        mothership=MothershipEnergyParameters(battery_capacity_wh=5.0),
        solar=SolarParameters(),
        timing=MissionTiming(max_deployment_cycles=2),
        propulsion_power_w=80.0,
    )

    assert result.mothership_energy_wh.min() >= 0.0
    assert result.quad_energy_wh.min() >= 0.0
    assert result.aborted
    assert result.states[-1] == MissionState.ABORT
