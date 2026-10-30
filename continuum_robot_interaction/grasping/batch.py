"""Independent PCC-controlled rod grasping worlds sharing one solver."""

import math

import newton
import numpy as np
import warp as wp

from .model import SEGMENT_LENGTHS
from .planning import GraspPlan
from .sim import BALL_RADIUS_RANGE, RAIL_HEIGHT_RANGE, Simulation, _Scene


class BatchSimulation(Simulation):
    """Run consecutive seeds in isolated worlds, displayed as a grid."""

    def __init__(self, count=576, seed=0, params=None, device=None, cup_position=None, *,
                 ball_radius_range=BALL_RADIUS_RANGE, rail_height_range=RAIL_HEIGHT_RANGE):
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError("count must be a positive integer")
        self.seed = int(seed) if seed is not None else int(np.random.randint(0, 2**16))
        self.seeds = np.arange(self.seed, self.seed + count, dtype=np.int64)
        self.device = wp.get_device(device)
        samples = wp.empty((count, int(GraspPlan.duration * 60) + 1, 3 * len(SEGMENT_LENGTHS)),
                           dtype=wp.float32, device=self.device)
        builder = newton.ModelBuilder(up_axis="Z", gravity=(0.0, 0.0, -9.81))
        scenes = []
        color_groups = []
        for world, scene_seed in enumerate(self.seeds):
            scene = _Scene(params=params, device=self.device, seed=int(scene_seed), cup_position=cup_position,
                           ball_radius_range=ball_radius_range, rail_height_range=rail_height_range)
            wp.copy(samples, scene.plan._targets, dest_offset=world * scene.plan._targets.size)
            # Retain per-scene access without retaining separate allocations.
            scene.plan._targets = samples[world]
            body_offset = builder.body_count
            shape_offset = builder.shape_count
            while len(color_groups) < len(scene.builder.body_color_groups):
                color_groups.append([])
            for group, bodies in zip(color_groups, scene.builder.body_color_groups):
                group.extend(np.asarray(bodies) + body_offset)
            builder.add_world(scene.builder, label_prefix=f"env_{world}")
            scene.base_body += body_offset
            scene.rod_bodies = [body + body_offset for body in scene.rod_bodies]
            scene.object_body += body_offset
            scene.cup_shapes = [shape + shape_offset for shape in scene.cup_shapes]
            del scene.builder
            scenes.append(scene)
        # Preserve each world's Gauss-Seidel order when merging independent worlds.
        builder.body_color_groups = [np.asarray(group, dtype=np.int32) for group in color_groups]
        self._initialize(builder, scenes, samples)

    def results(self):
        return self._results()


def run(count=576, params=None, device=None, seed=None, cup_position=None, *, headless=True,
        ball_radius_range=BALL_RADIUS_RANGE, rail_height_range=RAIL_HEIGHT_RANGE):
    """Simulate independent seeds together and display them in a square grid."""
    from ..main import run_simulation

    simulation = BatchSimulation(count=count, params=params, device=device, seed=seed, cup_position=cup_position,
                                 ball_radius_range=ball_radius_range, rail_height_range=rail_height_range)
    if headless:
        for _ in range(round(GraspPlan.duration / simulation.frame_dt)):
            simulation.step()
        return simulation.results()
    side = math.ceil(math.sqrt(count))
    distance = 1.4 * (0.85 * (side - 1) + 1.0)
    run_simulation(
        simulation, headless=False,
        world_spacing=(0.85, 0.85, 0.0),
        camera_position=(0.0, -0.5 * distance, 0.2 + math.sqrt(0.75) * distance),
        camera_pitch=-60.0, camera_yaw=90.0,
    )
    return simulation.results()
