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

...

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
