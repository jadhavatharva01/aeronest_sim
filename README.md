# AeroNest Simulation

AeroNest is a Python-first simulation workspace for validating a stacked-wing fixed-wing mothership before moving to heavier robotics tooling.

This repository intentionally starts with modular physics, configuration files, tests, and runnable phase scripts. ROS 2, Gazebo, and PX4 are out of scope until the Python model is validated.

## Repository Layout

```text
config/                  YAML vehicle, mission, and simulation parameters
src/aeronest/physics/    Flight physics models
src/aeronest/control/    Control models, future phases
src/aeronest/perception/ Perception models, future phases
src/aeronest/simulation/ Scenario orchestration, future phases
src/aeronest/visualization/ Plotting utilities
src/aeronest/utils/      Shared utilities, future phases
scripts/                 Runnable phase scripts
tests/                   Pytest tests
outputs/                 Generated plots and run artifacts
```

## Phase 1: Fixed-Wing Slow Flight

Phase 1 models the 1.5 m stacked-wing mothership using steady, level-flight approximations:

- Effective wing area includes a stacked-wing interference factor.
- Lift is computed from dynamic pressure, effective area, and lift coefficient.
- Drag includes parasite drag plus induced drag from Oswald efficiency and aspect ratio.
- Stall speed is based on weight, air density, effective area, and `C_Lmax`.
- Power required is drag power divided by propeller efficiency.
- Recommended docking speed is `1.25 * stall_speed`.

These are early feasibility calculations, not a full 6-DOF simulator. The model assumes steady air density, symmetric wing loading, and no unsteady aerodynamic effects from docking, propwash, gusts, or wing-stack wake interactions beyond the configured interference factor.

## Phase 2: Mission Energy Loop

Phase 2 adds a DJI Mini-class quadcopter model and a mission-level battery simulation:

- Quadcopter hover power uses ideal induced power over total rotor disk area.
- Electrical hover power includes motor/prop efficiency losses and avionics load.
- Flight power applies a maneuver factor and payload load.
- Battery capacity is tracked in watt-hours with a configurable usable fraction.
- Reserve-aware inspection time keeps a configured reserve fraction unused.
- Mothership energy rate includes fixed-wing propulsion, avionics, charging load, and solar assist.
- Solar power is an endurance assist term, not an infinite-flight assumption.

The Phase 2 mission state machine includes transit, loiter, deployment, inspection, return, assumed docking, charging, redeploy/return decision, and return-home states. Docking is intentionally modeled as a successful placeholder event; the actual docking controller and animation are reserved for Phase 3.

## Run

Use the existing virtual environment named `aeronest_env`; do not create a new one.

```bash
source aeronest_env/bin/activate
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest
python scripts/run_phase_1_fixed_wing.py
python scripts/run_phase_2_mission_energy.py
```

Phase 1 plots are written to `outputs/phase_1/`. Phase 2 plots are written to `outputs/phase_2/`.
