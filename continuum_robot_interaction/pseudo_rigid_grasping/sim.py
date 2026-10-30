"""Contact-only grasping with an imported tapered arm URDF and Newton physics."""

import colorsys
import math

import newton
import numpy as np
import warp as wp

from ..cli import DEFAULT_FRAME_DT, DEFAULT_SIM_PARAMS
from .model import ACTUATED_JOINT_NAMES, CUP_URDF_PATH, JOINT_NAMES, SECTION_SLICES, URDF_PATH, YAW_JOINT_NAME
from .planning import CUP_FLOOR_THICKNESS, CUP_FLOOR_Z, CUP_INNER_RADIUS, CUP_RADIUS_RANGE, CUP_SOURCE_CLEARANCE, CUP_WALL_HEIGHT, GraspPlan, OBJECT_CENTER

# Dimensions used for result checks must match assets/cup.urdf.
CUP_GEOMETRY = {"inner_radius": CUP_INNER_RADIUS, "wall_thickness": 0.008, "wall_height": CUP_WALL_HEIGHT,
                "floor_thickness": CUP_FLOOR_THICKNESS, "wall_count": 12}
CUP_RADIAL_RANGE = CUP_RADIUS_RANGE
BALL_RADIUS_RANGE = (0.015, 0.035)
RAIL_HEIGHT_RANGE = (0.120, 0.240)


def _geometry_range(values, name):
    bounds = np.asarray(values, dtype=float)
    if bounds.shape != (2,) or not np.isfinite(bounds).all() or not 0.0 < bounds[0] <= bounds[1]:
        raise ValueError(f"{name} must contain finite positive MIN <= MAX values in metres")
    return bounds


@wp.kernel
def joint_targets(
    curr_time: wp.array(dtype=wp.float64),
    time_offset: wp.float64,
    targets: wp.array2d(dtype=wp.float64),
    yaw_offset: wp.float64,
    indices: wp.array(dtype=int),
    output: wp.array(dtype=float),
):
    """Launch 25 threads to interpolate and write targets at control indices."""
    joint = wp.tid()
    sample = wp.clamp((curr_time[0] + time_offset) * wp.float64(60.0), wp.float64(0.0), wp.float64(targets.shape[0] - 1))
    lower = int(wp.floor(sample))
    upper = wp.min(lower + 1, targets.shape[0] - 1)
    alpha = sample - wp.float64(lower)
    target = targets[lower, joint] + alpha * (targets[upper, joint] - targets[lower, joint])
    if joint == 0:
        target -= yaw_offset
    output[indices[joint]] = float(target)

@wp.kernel
def _finish_frame(
    body_q: wp.array[wp.transform],
    object_body: int,
    frame_dt: wp.float64,
    curr_time: wp.array[wp.float64],
    max_object_height: wp.array[float],
):
    curr_time[0] += frame_dt
    height = wp.transform_get_translation(body_q[object_body])[2]
    max_object_height[0] = wp.max(max_object_height[0], height)


class _Scene:
    """Build one seeded scene before allocating physics state or a solver."""

    def __init__(self, params=None, device=None, seed=None, cup_position=None, *,
                 ball_radius_range=BALL_RADIUS_RANGE, rail_height_range=RAIL_HEIGHT_RANGE):
        self.params = DEFAULT_SIM_PARAMS.copy()
        self.params.update(params or {})
        assert self.params["substeps"] > 0 and isinstance(self.params["substeps"], int), "substeps must be a positive integer"
        assert self.params["iterations"] > 0 and isinstance(self.params["iterations"], int), "iterations must be a positive integer"
        assert self.params["substeps"] % 2 == 0, "substeps must be even for graph capture"
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
        builder.default_joint_cfg.target_ke = 100.0
        builder.add_urdf(str(URDF_PATH), floating=False, enable_self_collisions=True, joint_ordering="dfs")
        joint_labels = [label.rsplit("/", 1)[-1] for label in builder.joint_label]
        joint_indices = [joint_labels.index(name) for name in JOINT_NAMES]
        target_indices = [builder.joint_q_start[joint_labels.index(name)] for name in ACTUATED_JOINT_NAMES]
        yaw_joint = joint_labels.index(YAW_JOINT_NAME)
        # Center this joint's coordinates on the planned sweep, keeping the
        # native angular drive clear of its +/-pi representation boundary.
        self._yaw_offset = self.plan.target_yaw / 2
        builder.joint_X_p[yaw_joint] = wp.transform_multiply(
            builder.joint_X_p[yaw_joint],
            wp.transform(wp.vec3(0.0), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), self._yaw_offset)),
        )
        builder.joint_target_q[target_indices[0]] = -self._yaw_offset
        yaw_dof = builder.joint_qd_start[yaw_joint]
        builder.joint_target_ke[yaw_dof] = 3000.0
        builder.joint_target_kd[yaw_dof] = 10.0
        builder.joint_target_mode[yaw_dof] = newton.JointTargetMode.POSITION
        for section, stiffness, damping in zip(SECTION_SLICES, (3000.0, 1500.0, 300.0), (10.0, 5.0, 1.0)):
            for joint in joint_indices[section]:
                dof = builder.joint_qd_start[joint]
                builder.joint_target_ke[dof] = stiffness
                builder.joint_target_kd[dof] = damping
                builder.joint_target_mode[dof] = newton.JointTargetMode.POSITION
        # The first bending hinge pivots within the pedestal's mounting volume.
        first_link = builder.joint_child[joint_indices[0]]
        for base_shape, body in enumerate(builder.shape_body):
            if body == 0:
                for link_shape, link_body in enumerate(builder.shape_body):
                    if link_body == first_link:
                        builder.add_shape_collision_filter_pair(base_shape, link_shape)
        # The URDF pedestal is the fixed mounting, not a driven robot body.
        builder.body_mass[0] = builder.body_inv_mass[0] = 0.0
        builder.body_inertia[0] = builder.body_inv_inertia[0] = wp.mat33(0.0)

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

        # Collapse only the cup's fixed root, preserving the arm's joint indices.
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
        self.target_indices = target_indices


class Simulation(_Scene):
    """Command joint targets; contact and gravity determine every object pose."""

    def __init__(self, params=None, device=None, seed=None, cup_position=None, *,
                 ball_radius_range=BALL_RADIUS_RANGE, rail_height_range=RAIL_HEIGHT_RANGE):
        super().__init__(params=params, device=device, seed=seed, cup_position=cup_position,
                         ball_radius_range=ball_radius_range, rail_height_range=rail_height_range)
        builder = self.builder
        target_indices = self.target_indices
        self.time = 0.0
        self.curr_time = wp.zeros(1, dtype=wp.float64, device=self.device)
        self.frame_dt = DEFAULT_FRAME_DT
        self.model = builder.finalize(device=self.device)
        del self.builder
        # VBD measures angular strains relative to model.body_q, so its rest
        # pose must use that same centered frame. Initial physical poses still
        # point straight up at world yaw zero.
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.model)
        initial_coordinates = self.model.joint_q.numpy().copy()
        initial_coordinates[target_indices[0]] = -self._yaw_offset
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
        self._joint_indices = wp.array(target_indices, dtype=wp.int32, device=self.device)
        self._max_object_height = wp.array([self.object_initial_position[2]], dtype=float, device=self.device)
        self._initial_state = self.model.state()
        self._initial_state.assign(self.state)
        with wp.ScopedCapture(device=self.model.device) as capture:
            self._simulate()
        self.graph = capture.graph

    def reset(self):
        """Restart in place so the captured graph keeps its array references."""
        self.time = 0.0
        self.curr_time.zero_()
        self._max_object_height.fill_(self.object_initial_position[2])
        self.state.assign(self._initial_state)
        self._next_state.assign(self._initial_state)
        # Preserve the initial yaw correction while clearing solver history.
        self.solver.reset(self.state, flags=0)
        self.control.clear(self.model)
        self.collision_pipeline.reset_contact_matching()
        self.collision_pipeline.collide(self.state, self.contacts)

    def step(self):
        """Replay one captured physics frame."""
        wp.capture_launch(self.graph)
        self.time += self.frame_dt

    def _simulate(self):
        dt = self.frame_dt
        substeps = self.params["substeps"]
        with wp.ScopedDevice(self.device):
            wp.launch(
                joint_targets, dim=len(ACTUATED_JOINT_NAMES),
                inputs=[self.curr_time, dt, self.plan._targets, self._yaw_offset, self._joint_indices],
                outputs=[self.control.joint_target_q],
            )
            for _ in range(substeps):
                self.state.clear_forces()
                self.collision_pipeline.collide(self.state, self.contacts)
                self.solver.step(self.state, self._next_state, self.control, self.contacts, dt / substeps)
                self.state, self._next_state = self._next_state, self.state
            wp.launch(
                _finish_frame, dim=1,
                inputs=[self.state.body_q, self.object_body, dt],
                outputs=[self.curr_time, self._max_object_height],
            )

    def results(self):
        position = self.state.body_q.numpy()[self.object_body, :3].copy()
        lift = float(position[2] - self.object_initial_position[2])
        transport = float(np.linalg.norm(position[:2] - self.object_initial_position[:2]))
        target_error = float(np.linalg.norm(position[:2] - self.cup_position[:2]))
        speed = float(np.linalg.norm(self.state.body_qd.numpy()[self.object_body, :3]))
        floor = self.cup_position[2] + CUP_GEOMETRY["floor_thickness"] / 2
        rim = floor + CUP_GEOMETRY["wall_height"]
        wall_angles = 2 * math.pi * np.arange(CUP_GEOMETRY["wall_count"]) / CUP_GEOMETRY["wall_count"]
        wall_normals = np.column_stack((np.cos(wall_angles), np.sin(wall_angles)))
        wall_distance = np.max(wall_normals @ (position[:2] - self.cup_position[:2]))
        # A seated large ball may protrude above the rim.
        settled_top = max(rim - self.object_radius, floor + self.object_radius)
        in_cup = bool(wall_distance <= CUP_GEOMETRY["inner_radius"] - self.object_radius + 0.001
                      and floor + self.object_radius - 0.001 <= position[2] <= settled_top + 0.001)
        return {
            "object_initial_position": self.object_initial_position.tolist(),
            "object_final_position": position.tolist(),
            "lift_height": lift,
            "maximum_lift_height": max(0.0, float(self._max_object_height.numpy()[0] - self.object_initial_position[2])),
            "transport_distance": transport,
            "target_position_error": target_error,
            "object_speed": speed,
            "seed": self.seed,
            "ball_radius": self.object_radius,
            "rail_height": self.rail_height,
            "cup_position": self.cup_position.tolist(),
            "ball_in_cup": in_cup,
        }


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
