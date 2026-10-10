# Newton Continuum Robot Interactions

Extension of my [continuum robot interaction](https://github.com/MatthewAndreTaylor/continuum_robot_interaction)
work using [Newton](https://github.com/newton-physics/newton) physics and
[NVIDIA Warp](https://github.com/NVIDIA/warp).

- **Baseline:** a two-section arm interacting with a fixed cylinder.
- **Grasping:** a three-section arm picking up a randomly sized ball from support
  rails and placing it in a randomly positioned cup.

## Quickstart

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```sh
uv sync --locked
uv run continuum-robot baseline
uv run continuum-robot grasping
```

Each simulation opens an interactive desktop viewer. Add `--headless` to run
without a window, or `--device cpu` / `--device cuda:0` to select a device.

## Examples

### Grasping

Design inspiration: https://youtu.be/Yp2f9gQongk

I wanted to use three sections with rectangular cross-sections in a design better described as a hyper-redundant articulated manipulator.
The robot can still wrap around objects, and a piecewise constant curvature (PCC) approximation can still be used for motion planning.

#### Single environment, randomized pickup height, ball size, and goal cup location

https://github.com/user-attachments/assets/34f50667-8bb9-4c02-bb56-557f17d6c5fc

#### 5x5 annotated environments

https://github.com/user-attachments/assets/32eb47f4-138d-4d31-b1e2-65a0b4eccf40

#### 24x24 environments

https://github.com/user-attachments/assets/078f1ae7-4f27-4836-b557-8a689865e6a6

### Baseline

#### Forward kinematics and collision detection

https://github.com/user-attachments/assets/0c08c4e5-0f49-45d4-bf63-cace4474b69f

## Options

Repeat a scene with a fixed seed, or run multiple environments together:

```sh
uv run continuum-robot grasping --seed 7
uv run continuum-robot grasping --num-envs 576 --seed 0 --device cuda:0
```

Results report lift, transport, and whether the ball reached the cup. Use
`--help` for all options, including ball size, rail height, and cup position:

```sh
uv run continuum-robot baseline --help
uv run continuum-robot grasping --help
```
