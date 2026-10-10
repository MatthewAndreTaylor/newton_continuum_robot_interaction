"""Baseline: two-section PCC control with Newton's compliant rods and Warp kernels."""

import math

import newton
import numpy as np
import warp as wp

from ..cli import BASELINE_CONTROL_PARAMS, DEFAULT_FRAME_DT, DEFAULT_SIM_PARAMS

CYL_START = np.array([0, -0.70, 0.1])
CYL_DIRECTION = np.array([0.0, 1.0, 0.0])
CYL_LENGTH = 1.40
CYL_RADIUS = 0.03

@wp.func
def pcc_centerline(kappa: float, phi: float, s: float, start_T: wp.transform):
    """Return the world pose at arc length s along a section initially pointing -Y."""
    theta = kappa * s
    # Series and the half-angle identity avoid cancellation near a straight section.
    if wp.abs(theta) < 1.0e-3:
        theta2 = theta * theta
        axial = s * (1.0 - theta2 / 6.0)
        lateral = s * theta * (0.5 - theta2 / 24.0)
    else:
        axial = wp.sin(theta) / kappa
        half_sin = wp.sin(0.5 * theta)
        lateral = 2.0 * half_sin * half_sin / kappa
    axis = wp.vec3(wp.cos(phi), 0.0, wp.sin(phi))
    position = wp.vec3(axis[2] * lateral, -axial, -axis[0] * lateral)
    return wp.transform_multiply(start_T, wp.transform(position, wp.quat_from_axis_angle(axis, theta)))


@wp.func
def _pcc_robot_target(
    k1: float,
    phi1: float,
    k2: float,
    phi2: float,
    n_elem: int,
    base_length: float,
    body: int,
):
    element = body % n_elem
    kappa = k1
    phi = phi1
    start_T = wp.transform_identity()
    if body >= n_elem:
        start_T = pcc_centerline(k1, phi1, base_length, start_T)
        kappa = k2
        phi = phi2
    start = pcc_centerline(kappa, phi, float(element) * base_length / float(n_elem), start_T)
    end = pcc_centerline(kappa, phi, float(element + 1) * base_length / float(n_elem), start_T)
    return 0.5 * (wp.transform_get_translation(start) + wp.transform_get_translation(end))


@wp.kernel
def pcc_robot_targets(
    k1: float,
    phi1: float,
    k2: float,
    phi2: float,
    n_elem: int,
    base_length: float,
    targets: wp.array[wp.vec3],
):
    """Launch 2*n_elem threads to write section-ordered capsule chord midpoints."""
    body = wp.tid()
    targets[body] = _pcc_robot_target(k1, phi1, k2, phi2, n_elem, base_length, body)


@wp.kernel
def _update_targets(
    command: wp.vec4,
    frame: wp.array[int],
    ramp_step: float,
    n_elem: int,
    base_length: float,
    targets: wp.array[wp.vec3],
):
    body = wp.tid()
    fraction = wp.min(float(frame[0] + 1) * ramp_step, 1.0)
    targets[body] = _pcc_robot_target(
        command[0] * fraction, command[1], command[2] * fraction, command[3],
        n_elem, base_length, body,
    )


@wp.kernel
def _advance_frame(frame: wp.array[int]):
    frame[0] += 1


@wp.kernel
def _pcc_forces(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_inv_mass: wp.array[float],
    targets: wp.array[wp.vec3],
    gain: float,
    damping: float,
    dt: float,
    body_f: wp.array[wp.spatial_vector],
):
    body = wp.tid()
    if body_inv_mass[body] > 0.0:
        linear_velocity = wp.spatial_top(body_qd[body])
        error = targets[body] - wp.transform_get_translation(body_q[body])
        # Implicit local spring/damper update avoids explicit damping instability.
        kd = damping + gain * dt
        force = (gain * error - kd * linear_velocity) / (1.0 + kd * dt * body_inv_mass[body])
        body_f[body] = wp.spatial_vector(force, wp.vec3(0.0))


class Simulation:
    """Newton Cosserat rod approximation, with two independently driven sections.

    The first capsule is kinematic: it clamps the base position and director.
    Newton stores stretch/shear at joints between rigid capsule segments, so the
    first segment cannot stretch as a PyElastica element could.
    """

    n_elem = 10
    base_length = 0.25
    base_radius = 0.025

    def __init__(self, inputs, params=None, device=None):
        self.inputs = np.asarray(inputs, dtype=float)
        if self.inputs.shape != (4,) or not np.isfinite(self.inputs).all():
            raise ValueError("inputs must contain four finite values (kappa1, phi1, kappa2, phi2)")
        self.params = {**DEFAULT_SIM_PARAMS, **BASELINE_CONTROL_PARAMS}
        self.params.update(params)
        assert self.params["substeps"] > 0 and isinstance(self.params["substeps"], int), "substeps must be a positive integer"
        assert self.params["iterations"] > 0 and isinstance(self.params["iterations"], int), "iterations must be a positive integer"
        assert self.params["substeps"] % 2 == 0, "substeps must be even for graph capture"
        assert self.params["ramp_time"] > 0.0, "ramp_time must be positive"
        self.frame_dt = DEFAULT_FRAME_DT
        self.time = 0.0
        self.segment_length = self.base_length / self.n_elem
        self.device = wp.get_device(device)

        builder = newton.ModelBuilder(up_axis="Y", gravity=(0.0, 0.0, 0.0))
        cfg = newton.ModelBuilder.ShapeConfig(density=1000.0, ke=5e5, kd=5e2, mu=0.5, gap=0.005)
        points = np.zeros((2 * self.n_elem + 1, 3), dtype=np.float32)
        points[:, 1] = -np.arange(len(points)) * self.segment_length
        area = math.pi * self.base_radius**2
        second_moment = math.pi * self.base_radius**4 / 4.0
        youngs_modulus, shear_modulus = 1e7, 1e7 / 1.5
        rod = newton.Rod(
            points,
            radius=self.base_radius,
            stretch_rigidity=youngs_modulus * area,
            # PyElastica's circular-section shear correction is 27/28.
            shear_rigidity=(27.0 / 28.0) * shear_modulus * area,
            bend_rigidity=youngs_modulus * second_moment,
            twist_rigidity=shear_modulus * 2.0 * second_moment,
        )
        self.rod_bodies, _ = builder.add_rod(rod=rod, cfg=cfg, label="arm", body_frame_origin="com")

        # Anchor both position and material orientation at the arm's base.
        base = self.rod_bodies[0]
        builder.body_mass[base] = builder.body_inv_mass[base] = 0.0
        builder.body_inertia[base] = builder.body_inv_inertia[base] = wp.mat33(0.0)

        for body in self.rod_bodies:
            builder.shape_color[body] = (0.12, 0.45, 0.85) if body < self.n_elem else (0.95, 0.45, 0.12)

        # Keep rod self-collision disabled as in the original scene.
        for shape_a in range(len(self.rod_bodies)):
            for shape_b in range(shape_a + 1, len(self.rod_bodies)):
                builder.add_shape_collision_filter_pair(shape_a, shape_b)
                
        self.obstacle_shape = builder.add_shape_cylinder(
            body=-1,
            xform=wp.transform(
                wp.vec3(CYL_START + CYL_DIRECTION * CYL_LENGTH / 2),
                wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -math.pi / 2),
            ),
            radius=CYL_RADIUS,
            half_height=CYL_LENGTH / 2,
            cfg=cfg,
            color=(0.5, 0.5, 0.5),
            label="obstacle",
        )
        builder.add_ground_plane(height=float(CYL_START[1]), cfg=cfg)
        builder.color()
        self.model = builder.finalize(device=self.device)
        self.state = self.model.state()
        self._next_state = self.model.state()
        self.control = self.model.control()
        self.collision_pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.collision_pipeline.contacts()
        self.solver = newton.solvers.SolverVBD(
            self.model, iterations=int(self.params["iterations"]), rigid_compliant_alm=True
        )
        self.targets = wp.empty(2 * self.n_elem, dtype=wp.vec3, device=self.device)
        self._frame = wp.zeros(1, dtype=int, device=self.device)
        wp.launch(
            pcc_robot_targets,
            dim=2 * self.n_elem,
            inputs=[0.0, 0.0, 0.0, 0.0, self.n_elem, self.base_length],
            outputs=[self.targets],
            device=self.device,
        )
        self._initial_state = self.model.state()
        self._initial_state.assign(self.state)
        with wp.ScopedCapture(device=self.model.device) as capture:
            self._simulate()
        self.graph = capture.graph

    def reset(self):
        """Restart in place so the captured graph keeps its array references."""
        self.time = 0.0
        self._frame.zero_()
        self.state.assign(self._initial_state)
        self._next_state.assign(self._initial_state)
        self.solver.reset(self.state, flags=0)
        self.control.clear(self.model)
        self.collision_pipeline.reset_contact_matching()
        self.collision_pipeline.collide(self.state, self.contacts)

    def step(self):
        """Replay one captured physics frame."""
        wp.capture_launch(self.graph)
        self.time += self.frame_dt

    def _simulate(self):
        substeps = int(self.params["substeps"])
        sub_dt = self.frame_dt / substeps
        ramp_step = self.frame_dt / float(self.params["ramp_time"])
        with wp.ScopedDevice(self.device):
            wp.launch(
                _update_targets,
                dim=2 * self.n_elem,
                inputs=[wp.vec4(*self.inputs), self._frame, ramp_step,
                        self.n_elem, self.base_length],
                outputs=[self.targets],
            )
            for _ in range(substeps):
                wp.launch(
                    _pcc_forces,
                    dim=self.model.body_count,
                    inputs=[self.state.body_q, self.state.body_qd, self.model.body_inv_mass, self.targets,
                            float(self.params["gain"]), float(self.params["damping"]), sub_dt],
                    outputs=[self.state.body_f],
                )
                self.collision_pipeline.collide(self.state, self.contacts)
                self.solver.step(self.state, self._next_state, self.control, self.contacts, sub_dt)
                self.state, self._next_state = self._next_state, self.state
                
            wp.launch(_advance_frame, dim=1, inputs=[self._frame])

def run(inputs, params=None, device=None, *, headless=True):
    """Simulate with an optional live viewer and report any obstacle contact."""
    from ..main import run_simulation

    sim = Simulation(inputs, params=params, device=device)
    run_simulation(sim, headless=headless)
