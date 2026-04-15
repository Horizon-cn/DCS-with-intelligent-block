# DCS with Intelligent Block

This repository is organized as a monorepo with two mostly independent project
components.

## Components

- `dcs/`: PyBullet simulation, surface planning, and dynamic replanning code.
- `dynamixel_control/`: Dynamixel trajectory tooling and OpenRB-150 firmware.

Each component has its own README with setup and usage details.

## Layout

```text
dcs/
  README.md
  config.yaml
  main.py
  control/

dynamixel_control/
  README.md
  requirements.txt
  environment.yml
  firmware/
```
