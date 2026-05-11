Project: AeroNest UAV mothership simulation.

Work style:
- Build phase by phase.
- Start with Python-only physics simulation.
- Do not jump to ROS 2, Gazebo, or PX4 until the Python model is validated.
- Use modular files, tests, and YAML configs.
- Every phase must have a runnable script.
- Save plots to outputs/.
- Use numpy, scipy, matplotlib, pandas, pyyaml, pytest.
- Use dataclasses for vehicle parameters.
- Do not hardcode physical constants if they belong in config.
- After every major change, run pytest and the relevant phase script.
- Explain assumptions and simplifications in README.md.

Physics priorities:
- Fixed-wing stall speed, lift, drag, and power.
- Stacked-wing interference factor.
- Quadcopter hover power and battery model.
- Docking relative kinematics.
- Wind disturbance.
- Small AprilTag detection range limits.
- Energy feasibility.

Important realism constraints:
- Finger-sized AprilTags are only for final close-range alignment.
- Solar is endurance assist, not guaranteed infinite flight.
- Recharging is not instant.
- Docking at 6.5–8 m/s is ambitious and should be tested through Monte Carlo simulation.
