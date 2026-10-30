"""Three elastic rod sections grasping a ball through contact forces."""

import colorsys
import math

import newton
import numpy as np
import warp as wp

from ..cli import BASELINE_CONTROL_PARAMS, DEFAULT_FRAME_DT, DEFAULT_SIM_PARAMS
from .model import CUP_URDF_PATH, SEGMENT_LENGTHS, add_arm
from .control import pcc_forces, update_targets
from .planning import CUP_FLOOR_THICKNESS, CUP_FLOOR_Z, CUP_INNER_RADIUS, CUP_RADIUS_RANGE, CUP_SOURCE_CLEARANCE, CUP_WALL_HEIGHT, GraspPlan, OBJECT_CENTER

# Dimensions used for result checks must match assets/cup.urdf.
CUP_GEOMETRY = {"inner_radius": CUP_INNER_RADIUS, "wall_thickness": 0.008, "wall_height": CUP_WALL_HEIGHT,
                "floor_thickness": CUP_FLOOR_THICKNESS, "wall_count": 12}
CUP_RADIAL_RANGE = CUP_RADIUS_RANGE
BALL_RADIUS_RANGE = (0.015, 0.035)
RAIL_HEIGHT_RANGE = (0.120, 0.240)
GRASP_CONTROL_PARAMS = {"gain": BASELINE_CONTROL_PARAMS["gain"], "damping": 450.0}


def _geometry_range(values, name):
    bounds = np.asarray(values, dtype=float)
    if bounds.shape != (2,) or not np.isfinite(bounds).all() or not 0.0 < bounds[0] <= bounds[1]:
        raise ValueError(f"{name} must contain finite positive MIN <= MAX values in metres")
    return bounds


class _Scene:
    """Build one seeded scene before allocating physics state or a solver."""

    def __init__(self, params=None, device=None, seed=None, cup_position=None, *,
                 ball_radius_range=BALL_RADIUS_RANGE, rail_height_range=RAIL_HEIGHT_RANGE):
        self.params = {**DEFAULT_SIM_PARAMS, **GRASP_CONTROL_PARAMS}
        self.params.update(params or {})
        for name in ("substeps", "iterations"):
            value = self.params[name]
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.params["substeps"] % 2:
            raise ValueError("substeps must be even for graph capture")
        for name in ("gain", "damping"):
            if not math.isfinite(self.params[name]) or self.params[name] < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        ball_radius_range = _geometry_range(ball_radius_range, "ball_radius_range")
        rail_height_range = _geometry_range(rail_height_range, "rail_height_range")
        if ball_radius_range[1] >= CUP_INNER_RADIUS:
            raise ValueError("ball_radius_range must fit inside the cup's 0.055 m inner radius")
        self.seed = np.random.randint(0, 2**16) if seed is None else int(seed)
        rng = np.random.default_rng(self.seed)
        self.object_radius = float(rng.uniform(*ball_radius_range))
        self.rail_height = float(rng.uniform(*rail_height_range))
        if cup_position is None:
            while True:
                radius = math.sqrt(rng.uniform(CUP_RADIAL_RANGE[0]**2, CUP_RADIAL_RANGE[1]**2))
                azimuth = rng.uniform(-math.pi, math.pi)
                cup_position = [radius * math.cos(azimuth), radius * math.sin(azimuth), CUP_FLOOR_Z]
                if np.linalg.norm(np.asarray(cup_position[:2]) - OBJECT_CENTER[:2]) >= CUP_SOURCE_CLEARANCE:
                    break
        self.cup_position = np.asarray(cup_position, dtype=float)
        if (self.cup_position.shape != (3,) or not np.isfinite(self.cup_position).all()
                or not CUP_RADIAL_RANGE[0] - 1e-9 <= np.linalg.norm(self.cup_position[:2]) <= CUP_RADIAL_RANGE[1] + 1e-9
                or not math.isclose(self.cup_position[2], CUP_FLOOR_Z, abs_tol=1e-9)
                or np.linalg.norm(self.cup_position[:2] - OBJECT_CENTER[:2]) < CUP_SOURCE_CLEARANCE):
            raise ValueError("cup_position must be on the ground, 0.18-0.29 m from the base and clear of the pickup cradle")
        self.object_color = colorsys.hsv_to_rgb(rng.uniform(0.0, 1.0), 0.75, 0.95)
        self.device = wp.get_device(device)
        self.plan = GraspPlan(ball_radius=self.object_radius, rail_height=self.rail_height,
                              cup_position=self.cup_position, device=self.device)
        self.object_initial_position = np.array(self.plan.object_center, dtype=float)

        builder = newton.ModelBuilder(up_axis="Z", gravity=(0.0, 0.0, -9.81))
        builder.default_shape_cfg.ke = 2.0e5
        builder.default_shape_cfg.kd = 100.0
        builder.default_shape_cfg.mu = 1.0
        builder.default_shape_cfg.gap = 0.002
        self.base_body, self.rod_bodies = add_arm(builder)

        surface = newton.ModelBuilder.ShapeConfig(density=0.0, ke=2e5, kd=100.0, mu=1.0, gap=0.002)
        builder.add_ground_plane(cfg=surface)
        # The cradle supports the sphere while leaving room for the distal wrap underneath.
        rail_half_width = 0.006
        rail_offset = 0.8 * self.object_radius + rail_half_width
        self.rail_offset = rail_offset
        for side in (-1.0, 1.0):
            builder.add_shape_box(
                -1,
                xform=wp.transform(wp.vec3(self.object_initial_position[0], side * rail_offset, self.rail_height / 2), wp.quat_identity()),
                hx=0.015, hy=rail_half_width, hz=self.rail_height / 2,
                cfg=surface, color=(0.45, 0.47, 0.51), label=f"cradle_{side:g}",
            )

        # Import the shared cup as static collision geometry.
        cup_builder = newton.ModelBuilder(up_axis="Z")
        cup_builder.default_shape_cfg = surface
        cup_builder.add_urdf(
            str(CUP_URDF_PATH), floating=False, collapse_fixed_joints=True,
            parse_visuals_as_colliders=True,
        )
        cup_start = builder.shape_count
        builder.add_builder(
            cup_builder, xform=wp.transform(wp.vec3(self.cup_position), wp.quat_identity()), label_prefix="cup",
        )
        self.cup_shapes = list(range(cup_start, builder.shape_count))

        self.object_body = builder.add_body(
            xform=wp.transform(wp.vec3(self.object_initial_position), wp.quat_identity()), label="grasp_object"
        )
        object_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.04 / ((4.0 / 3.0) * math.pi * self.object_radius**3),
            ke=2e5, kd=100.0, mu=1.0, gap=0.002,
        )
        self.object_shape = builder.add_shape_sphere(
            self.object_body, radius=self.object_radius, cfg=object_cfg,
            color=self.object_color, label="grasp_object_collision",
        )
        builder.color()
        self.builder = builder


@wp.kernel
def _finish_frame(
    body_q: wp.array[wp.transform],
    object_bodies: wp.array[int],
    frame_dt: wp.float64,
    curr_time: wp.array[wp.float64],
    max_object_height: wp.array[float],
):
    world = wp.tid()
    if world == 0:
        curr_time[0] += frame_dt
    height = wp.transform_get_translation(body_q[object_bodies[world]])[2]
    max_object_height[world] = wp.max(max_object_height[world], height)


class Simulation(_Scene):
    """Drive rod centerlines with PCC forces; objects move by contact."""

    def __init__(self, params=None, device=None, seed=None, cup_position=None, *,
                 ball_radius_range=BALL_RADIUS_RANGE, rail_height_range=RAIL_HEIGHT_RANGE):
        super().__init__(params=params, device=device, seed=seed, cup_position=cup_position,
                         ball_radius_range=ball_radius_range, rail_height_range=rail_height_range)
        builder = self.builder
        del self.builder
        self._initialize(builder, [self], self.plan._targets.reshape((1, *self.plan._targets.shape)))

    def _initialize(self, builder, scenes, samples):
        self.scenes = scenes
        self.count = len(scenes)
        self.params = scenes[0].params
        self.time = 0.0
        self.frame_dt = DEFAULT_FRAME_DT
        self.curr_time = wp.zeros(1, dtype=wp.float64, device=self.device)
        self.model = builder.finalize(device=self.device)
        self.state = self.model.state()
        self._next_state = self.model.state()
        self.control = self.model.control()
        self.collision_pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.collision_pipeline.contacts()
        self.solver = newton.solvers.SolverVBD(
            self.model, iterations=self.params["iterations"], rigid_compliant_alm=True,
        )
        self._samples = samples
        self._rod_bodies = wp.array([body for scene in scenes for body in scene.rod_bodies],
                                    dtype=int, device=self.device)
        self.targets = wp.empty(self._rod_bodies.size, dtype=wp.vec3, device=self.device)
        self.target_velocities = wp.empty_like(self.targets)
        self.object_bodies = np.asarray([scene.object_body for scene in scenes], dtype=np.int32)
        self._object_bodies = wp.array(self.object_bodies, dtype=int, device=self.device)
        self._initial_heights = wp.array([scene.object_initial_position[2] for scene in scenes], dtype=float, device=self.device)
        self._max_object_height = wp.clone(self._initial_heights)
        self._initial_state = self.model.state()
        self._initial_state.assign(self.state)
        with wp.ScopedCapture(device=self.device) as capture:
            self._simulate()
        self.graph = capture.graph
        self.reset()

    def reset(self):
        """Restore the rods, ball, controller clock and solver history in place."""
        self.time = 0.0
        self.curr_time.zero_()
        wp.copy(self._max_object_height, self._initial_heights)
        self.state.assign(self._initial_state)
        self._next_state.assign(self._initial_state)
        self.solver.reset(self.state, flags=0)
        self.control.clear(self.model)
        self.collision_pipeline.reset_contact_matching()
        self.collision_pipeline.collide(self.state, self.contacts)
        wp.launch(update_targets, dim=(self.count, len(SEGMENT_LENGTHS)),
                  inputs=[self.curr_time, 0.0, self._samples],
                  outputs=[self.targets, self.target_velocities], device=self.device)

    def step(self):
        wp.capture_launch(self.graph)
        self.time += self.frame_dt

    def _simulate(self):
        substeps = self.params["substeps"]
        dt = self.frame_dt / substeps
        with wp.ScopedDevice(self.device):
            for substep in range(substeps):
                wp.launch(update_targets, dim=(self.count, len(SEGMENT_LENGTHS)),
                          inputs=[self.curr_time, (substep + 1) * dt, self._samples],
                          outputs=[self.targets, self.target_velocities])
                self.state.clear_forces()
                wp.launch(pcc_forces, dim=self._rod_bodies.size,
                          inputs=[self.state.body_q, self.state.body_qd, self.model.body_inv_mass,
                                  self._rod_bodies, self.targets, self.target_velocities,
                                  self.params["gain"], self.params["damping"], dt],
                          outputs=[self.state.body_f])
                self.collision_pipeline.collide(self.state, self.contacts)
                self.solver.step(self.state, self._next_state, self.control, self.contacts, dt)
                self.state, self._next_state = self._next_state, self.state
            wp.launch(_finish_frame, dim=self.count,
                      inputs=[self.state.body_q, self._object_bodies, self.frame_dt],
                      outputs=[self.curr_time, self._max_object_height])

    def _results(self):
        poses = self.state.body_q.numpy()
        velocities = self.state.body_qd.numpy()
        heights = self._max_object_height.numpy()
        wall_angles = 2 * math.pi * np.arange(CUP_GEOMETRY["wall_count"]) / CUP_GEOMETRY["wall_count"]
        normals = np.column_stack((np.cos(wall_angles), np.sin(wall_angles)))
        results = []
        for world, scene in enumerate(self.scenes):
            position = poses[scene.object_body, :3]
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
                "maximum_lift_height": max(0.0, float(heights[world] - scene.object_initial_position[2])),
                "transport_distance": float(np.linalg.norm(position[:2] - scene.object_initial_position[:2])),
                "target_position_error": float(np.linalg.norm(position[:2] - scene.cup_position[:2])),
                "object_speed": float(np.linalg.norm(velocities[scene.object_body, :3])),
                "ball_in_cup": bool(in_cup),
            })
        return results

    def results(self):
        return self._results()[0]


def run(params=None, device=None, seed=None, cup_position=None, *, headless=True,
        ball_radius_range=BALL_RADIUS_RANGE, rail_height_range=RAIL_HEIGHT_RANGE):
    """Simulate pickup, transport, and release, optionally in a live viewer."""
    from ..main import run_simulation
    from .measurements import draw_measurements

    simulation = Simulation(params=params, device=device, seed=seed, cup_position=cup_position,
                            ball_radius_range=ball_radius_range, rail_height_range=rail_height_range)
    if headless:
        for _ in range(round(GraspPlan.duration / simulation.frame_dt)):
            simulation.step()
        return simulation.results()
    run_simulation(
        simulation, headless=False,
        camera_position=(0.25, -0.65, 1.5), camera_pitch=-67.0, camera_yaw=120.0,
        draw_ui=draw_measurements,
    )
    return simulation.results()
