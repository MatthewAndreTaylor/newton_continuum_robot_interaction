"""Independent seeded grasping worlds sharing one solver and captured frame."""

import math

import newton
import numpy as np
import warp as wp

from ..cli import DEFAULT_FRAME_DT
from .model import ACTUATED_JOINT_NAMES
from .planning import GraspPlan
from .sim import BALL_RADIUS_RANGE, CUP_GEOMETRY, RAIL_HEIGHT_RANGE, Simulation, _Scene


@wp.kernel
def _joint_targets(
    curr_time: wp.array(dtype=wp.float64),
    time_offset: wp.float64,
    targets: wp.array3d(dtype=wp.float64),
    yaw_offsets: wp.array(dtype=wp.float64),
    indices: wp.array2d(dtype=int),
    output: wp.array(dtype=float),
):
    world, joint = wp.tid()
    sample = wp.clamp((curr_time[0] + time_offset) * wp.float64(60.0), wp.float64(0.0), wp.float64(targets.shape[1] - 1))
    lower = int(wp.floor(sample))
    upper = wp.min(lower + 1, targets.shape[1] - 1)
    alpha = sample - wp.float64(lower)
    target = targets[world, lower, joint] + alpha * (targets[world, upper, joint] - targets[world, lower, joint])
    if joint == 0:
        target -= yaw_offsets[world]
    output[indices[world, joint]] = float(target)


@wp.kernel
def _finish_frame(
    body_q: wp.array(dtype=wp.transform),
    object_bodies: wp.array(dtype=int),
    frame_dt: wp.float64,
    curr_time: wp.array(dtype=wp.float64),
    max_object_height: wp.array(dtype=float),
):
    world = wp.tid()
    if world == 0:
        curr_time[0] += frame_dt
    height = wp.transform_get_translation(body_q[object_bodies[world]])[2]
    max_object_height[world] = wp.max(max_object_height[world], height)


class BatchSimulation(Simulation):
    """Run seeds ``seed .. seed + count - 1`` in isolated, local-coordinate worlds.

    Viewer world offsets arrange the independent worlds into a visible grid;
    physics coordinates stay near the origin for identical numeric precision.
    """

    def __init__(self, count=576, seed=0, params=None, device=None, cup_position=None, *,
                 ball_radius_range=BALL_RADIUS_RANGE, rail_height_range=RAIL_HEIGHT_RANGE):
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError("count must be a positive integer")
        self.count = count
        self.seed = int(seed) if seed is not None else int(np.random.randint(0, 2**16))
        self.seeds = np.arange(self.seed, self.seed + count, dtype=np.int64)
        self.device = wp.get_device(device)
        self.time = 0.0
        self.frame_dt = DEFAULT_FRAME_DT
        self.curr_time = wp.zeros(1, dtype=wp.float64, device=self.device)
        self._targets = wp.empty((count, int(GraspPlan.duration * 60) + 1, len(ACTUATED_JOINT_NAMES)),
                                 dtype=wp.float64, device=self.device)
        builder = newton.ModelBuilder(up_axis="Z", gravity=(0.0, 0.0, -9.81))
        self.scenes = []
        object_bodies = []
        target_indices = []
        color_groups = []
        for world, scene_seed in enumerate(self.seeds):
            scene = _Scene(params=params, device=self.device, seed=int(scene_seed), cup_position=cup_position,
                           ball_radius_range=ball_radius_range, rail_height_range=rail_height_range)
            object_bodies.append(builder.body_count + scene.object_body)
            target_indices.append(np.asarray(scene.target_indices) + builder.joint_coord_count)
            if not color_groups:
                color_groups = [[] for _ in scene.builder.body_color_groups]
            for group, bodies in zip(color_groups, scene.builder.body_color_groups):
                group.extend(np.asarray(bodies) + builder.body_count)
            builder.add_world(scene.builder, label_prefix=f"env_{world}")
            wp.copy(self._targets, scene.plan._targets, dest_offset=world * scene.plan._targets.size)
            scene.plan._targets = self._targets[world]
            del scene.builder
            self.scenes.append(scene)
        # Preserve each world's Gauss-Seidel order instead of balancing groups across worlds.
        builder.body_color_groups = [np.asarray(group, dtype=np.int32) for group in color_groups]
        self.params = self.scenes[0].params
        self.object_bodies = np.asarray(object_bodies, dtype=np.int32)
        self.object_radii = np.asarray([scene.object_radius for scene in self.scenes])
        self.rail_heights = np.asarray([scene.rail_height for scene in self.scenes])
        self.cup_positions = np.asarray([scene.cup_position for scene in self.scenes])
        self.object_initial_positions = np.asarray([scene.object_initial_position for scene in self.scenes])
        self._object_bodies = wp.array(self.object_bodies, dtype=int, device=self.device)
        self._joint_indices = wp.array(np.asarray(target_indices), dtype=int, device=self.device)
        yaw_offsets = np.asarray([scene._yaw_offset for scene in self.scenes])
        self._yaw_offsets = wp.array(yaw_offsets, dtype=wp.float64, device=self.device)
        self.model = builder.finalize(device=self.device)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.model)
        initial_coordinates = self.model.joint_q.numpy().copy()
        initial_coordinates[np.asarray(target_indices)[:, 0]] = -yaw_offsets
        initial_coordinates = wp.array(initial_coordinates, dtype=float, device=self.device)
        self.state = self.model.state()
        self._next_state = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, initial_coordinates, self.model.joint_qd, self.state)
        newton.eval_fk(self.model, initial_coordinates, self.model.joint_qd, self._next_state)
        self.collision_pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.collision_pipeline.contacts()
        self.solver = newton.solvers.SolverVBD(
            self.model, iterations=int(self.params["iterations"]), rigid_compliant_alm=True,
            rigid_joint_linear_ke=1e6, rigid_joint_angular_ke=1e6,
        )
        self._initial_heights = wp.array(self.object_initial_positions[:, 2], dtype=float, device=self.device)
        self._max_object_height = wp.clone(self._initial_heights)
        self._initial_state = self.model.state()
        self._initial_state.assign(self.state)
        with wp.ScopedCapture(device=self.device) as capture:
            self._simulate()
        self.graph = capture.graph

    def reset(self):
        self.time = 0.0
        self.curr_time.zero_()
        wp.copy(self._max_object_height, self._initial_heights)
        self.state.assign(self._initial_state)
        self._next_state.assign(self._initial_state)
        self.solver.reset(self.state, flags=0)
        self.control.clear(self.model)
        self.collision_pipeline.reset_contact_matching()
        self.collision_pipeline.collide(self.state, self.contacts)

    def _simulate(self):
        substeps = self.params["substeps"]
        with wp.ScopedDevice(self.device):
            wp.launch(_joint_targets, dim=(self.count, len(ACTUATED_JOINT_NAMES)),
                      inputs=[self.curr_time, self.frame_dt, self._targets, self._yaw_offsets, self._joint_indices],
                      outputs=[self.control.joint_target_q])
            for _ in range(substeps):
                self.state.clear_forces()
                self.collision_pipeline.collide(self.state, self.contacts)
                self.solver.step(self.state, self._next_state, self.control, self.contacts, self.frame_dt / substeps)
                self.state, self._next_state = self._next_state, self.state
            wp.launch(_finish_frame, dim=self.count,
                      inputs=[self.state.body_q, self._object_bodies, self.frame_dt],
                      outputs=[self.curr_time, self._max_object_height])

    def results(self):
        positions = self.state.body_q.numpy()[self.object_bodies, :3]
        speeds = np.linalg.norm(self.state.body_qd.numpy()[self.object_bodies, :3], axis=1)
        maximum_lifts = self._max_object_height.numpy() - self.object_initial_positions[:, 2]
        wall_angles = 2 * math.pi * np.arange(CUP_GEOMETRY["wall_count"]) / CUP_GEOMETRY["wall_count"]
        normals = np.column_stack((np.cos(wall_angles), np.sin(wall_angles)))
        results = []
        for world, scene in enumerate(self.scenes):
            position = positions[world]
            floor = scene.cup_position[2] + CUP_GEOMETRY["floor_thickness"] / 2
            rim = floor + CUP_GEOMETRY["wall_height"]
            wall_distance = np.max(normals @ (position[:2] - scene.cup_position[:2]))
            in_cup = (wall_distance <= CUP_GEOMETRY["inner_radius"] - scene.object_radius + 0.001
                      and floor + scene.object_radius - 0.001 <= position[2]
                      <= max(rim - scene.object_radius, floor + scene.object_radius) + 0.001)
            results.append({
                "seed": scene.seed,
                "ball_radius": scene.object_radius,
                "rail_height": scene.rail_height,
                "cup_position": scene.cup_position.tolist(),
                "object_initial_position": scene.object_initial_position.tolist(),
                "object_final_position": position.tolist(),
                "lift_height": float(position[2] - scene.object_initial_position[2]),
                "maximum_lift_height": max(0.0, float(maximum_lifts[world])),
                "transport_distance": float(np.linalg.norm(position[:2] - scene.object_initial_position[:2])),
                "target_position_error": float(np.linalg.norm(position[:2] - scene.cup_position[:2])),
                "object_speed": float(speeds[world]),
                "ball_in_cup": bool(in_cup),
            })
        return results


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
