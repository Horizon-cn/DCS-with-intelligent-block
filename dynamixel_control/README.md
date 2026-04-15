# Dynamixel Control

Python trajectory tooling and OpenRB-150 firmware for a 5-DOF Dynamixel inchworm robot.

This component is organized around one main workflow:

1. Write a trajectory as hardware joint waypoints or task-space keyframes.
2. Optionally generate a JSON trajectory from task-space keyframes.
3. Execute the trajectory through the OpenRB-150 serial interface.

## Setup

Create the Conda environment:

```powershell
conda env create -f environment.yml
conda activate dynamixel-control
```

Or install the Python dependencies into an existing environment:

```powershell
pip install -r requirements.txt
```

The OpenRB-150 firmware lives in `firmware/firmware.ino`. Flash that sketch to the board before running trajectories from Python.

## Trajectory Format

Executable trajectories are lists of waypoints:

```python
[
    [j1, j2, j3, j4, j5],
    [j1, j2, j3, j4, j5, pwm1, pwm2],
]
```

The first five values are Dynamixel joint angles in degrees. The optional sixth and seventh values are PWM servo angles for gripper channels 1 and 2. Use `None` or omit the PWM values to leave a gripper unchanged at that step.

Example hardware-space trajectory:

```python
trajectory = [
    [180, 224.6, 86.95, 222.6, 180, 100, 100],
    [180, 224.6, 86.95, 222.6, 180, 100, 180],
    [180, 182.0, 77.7, 254.7, 180],
    [180, 182.6, 3.5, 180.6, 180, 100, 100],
]
```

You can store this in a Python file using any of these variable names:

```python
trajectory = [
    [180, 224.6, 86.95, 222.6, 180, 100, 100],
    [180, 182.0, 77.7, 254.7, 180],
]
```

Recognized names are `trajectory`, `TRAJECTORY`, `path`, `PATH`, `waypoints`, or `WAYPOINTS`. You can also pass a different variable name with `--variable`.

## Execute A Hardware-Space Trajectory

Run a Python trajectory file:

```powershell
python -m trajectories.entry path\to\my_trajectory.py --port COM8 --delay 0.25
```

Before moving the robot, `trajectories.entry` resets the Dynamixels, applies firmware-side configuration, pings IDs `1` through `5`, enables torque, executes the path, then disables torque on exit.

Useful options:

```powershell
python -m trajectories.entry path\to\my_trajectory.py --debug
python -m trajectories.entry path\to\my_trajectory.py --readback
python -m trajectories.entry path\to\my_trajectory.py --skip-init
python -m trajectories.entry path\to\my_trajectory.py --variable MY_TRAJECTORY
```

`--debug` loads and converts the trajectory without sending anything to hardware.

## Write A Task-Space Trajectory

For task-space authoring, edit `build_keyframes()` in `trajectories/generate_task_space_trajectory.py`.

Each row is:

```python
[x, y, z, forward_axis, pwm1, pwm2]
```

After the first row, you can omit `pwm1` and `pwm2` to keep the previous gripper state:

```python
rows = [
    [85, 0, 0, (-1, 0, 0), 100, 100],
    [85, 0, 35, (-1, 0, 0)],
    [170, 0, 35, (-1, 0, 0)],
    [170, 0, 0, (-1, 0, 0), 180, 100],
]
```

The generator interpolates between keyframes, solves IK using `kinematics_backend/analytical_ik.py`, and writes an IK-space JSON trajectory.

Generate the JSON:

```powershell
python -m trajectories.generate_task_space_trajectory --output generated\task_space_trajectory.json --samples-per-segment 10
```

Preview the generated trajectory without moving hardware:

```powershell
python -m trajectories.entry generated\task_space_trajectory.json --debug
```

Execute it:

```powershell
python -m trajectories.entry generated\task_space_trajectory.json --port COM8 --delay 0.25
```

JSON inputs default to `--source-frame ik`, so they are converted through `robot_config.profile.ik_to_hardware_angles()` before execution. Python trajectory files default to hardware-space angles. Override this when needed:

```powershell
python -m trajectories.entry my_path.json --source-frame hardware
python -m trajectories.entry my_path.py --source-frame ik
```

## Debugging Helpers

Solve and inspect one IK target:

```powershell
python -m tools.ik_debug --position 100,0,20 --forward-axis=-1,0,0 --text-only
```

Visualize the current robot profile:

```powershell
python -m tools.visualize_robot --angles 180,224.6,86.95,222.6,180
```

Set a PWM channel or read current Dynamixel positions:

```powershell
python -m control.pwm_cli 1 90 --port COM8
python -m control.pwm_cli --readback --port COM8
```

## Important Files

- `trajectories/entry.py`: loads and executes `.py` or `.json` trajectories.
- `trajectories/generate_task_space_trajectory.py`: task-space keyframe authoring and JSON generation.
- `trajectories/helpers.py`: waypoint normalization and interpolation helpers.
- `robot_config/profile.py`: measured robot geometry and IK-to-hardware angle conversion.
- `kinematics_backend/`: IK, robot definition, path building, and visualization helpers.
- `control/openRB150interface.py`: host-side serial interface for the OpenRB-150 command protocol.
- `control/pwm_cli.py`: quick PWM and Dynamixel readback utility.
- `tools/`: IK and geometry debugging CLIs.
- `firmware/firmware.ino`: robot-side OpenRB-150 firmware command interpreter.
