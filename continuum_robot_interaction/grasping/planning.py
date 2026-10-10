"""A staged PCC grasp and cup deposit with a rotating base.

The previous IK solution seeds the next one, following the motion-history idea
in OpenTDCRContactModel (https://github.com/ContinuumRoboticsLab/OpenTDCRContactModel).
This is geometric planning; Newton resolves the resulting object contacts.
"""

import math

import numpy as np
import warp as wp

from .model import BASE_HEIGHT, GRIP_RIDGE_HEIGHT, LINK_LENGTHS, LINK_THICKNESSES, SECTION_COUNTS


OBJECT_CENTER = np.array([0.17, 0.0, 0.25])
OBJECT_RADIUS = 0.025
CUP_RADIUS_RANGE = (0.18, 0.29)
CUP_SOURCE_CLEARANCE = 0.100
CUP_FLOOR_Z = 0.004
CUP_INNER_RADIUS = 0.055
CUP_WALL_HEIGHT = 0.06
CUP_FLOOR_THICKNESS = 0.008

@wp.func
def _smoothstep(value: wp.float64):
    value = wp.clamp(value, wp.float64(0.0), wp.float64(1.0))
    return value * value * (wp.float64(3.0) - wp.float64(2.0) * value)


@wp.func
def _grasp_center(proximal: wp.vec2d, distal_angle: wp.float64, circle_radius: wp.float64, straight_count: int):
    orientation = wp.float64(0.0)
    center = wp.vec2d(wp.float64(0.0), wp.float64(BASE_HEIGHT))
    for index in range(wp.static(sum(SECTION_COUNTS[:2]))):
        angle = proximal[0] / wp.float64(wp.static(SECTION_COUNTS[0]))
        length = wp.float64(wp.static(LINK_LENGTHS[0]))
        if index >= wp.static(SECTION_COUNTS[0]):
            angle = proximal[1] / wp.float64(wp.static(SECTION_COUNTS[1]))
            length = wp.float64(wp.static(LINK_LENGTHS[SECTION_COUNTS[0]]))
        orientation += angle
        center += length * wp.vec2d(wp.sin(orientation), wp.cos(orientation))
    center += wp.float64(straight_count) * wp.float64(wp.static(LINK_LENGTHS[-1])) * wp.vec2d(wp.sin(orientation), wp.cos(orientation))
    heading = orientation - distal_angle / wp.float64(2.0)
    return center + circle_radius * wp.vec2d(wp.cos(heading), -wp.sin(heading))


@wp.func
def _solve_center(target: wp.vec2d, initial: wp.vec2d, distal_angle: wp.float64, circle_radius: wp.float64, straight_count: int):
    angles = initial
    for _ in range(100):
        current = _grasp_center(angles, distal_angle, circle_radius, straight_count)
        error = target - current
        if wp.length(error) < wp.float64(1e-8):
            return angles, True
        delta = wp.float64(1e-5)
        dx = (_grasp_center(angles + wp.vec2d(delta, wp.float64(0.0)), distal_angle, circle_radius, straight_count) - current) / delta
        dy = (_grasp_center(angles + wp.vec2d(wp.float64(0.0), delta), distal_angle, circle_radius, straight_count) - current) / delta
        # Solve the damped 2x2 normal equation without a host-side NumPy solve.
        a = dx[0] * dx[0] + dy[0] * dy[0] + wp.float64(1e-6)
        b = dx[0] * dx[1] + dy[0] * dy[1]
        d = dx[1] * dx[1] + dy[1] * dy[1] + wp.float64(1e-6)
        projected = wp.vec2d(d * error[0] - b * error[1], a * error[1] - b * error[0]) / (a * d - b * b)
        step = wp.vec2d(wp.dot(dx, projected), wp.dot(dy, projected))
        step *= wp.min(wp.float64(1.0), wp.float64(0.25) / wp.max(wp.length(step), wp.float64(1e-12)))
        angles = wp.vec2d(
            wp.clamp(angles[0] + step[0], wp.float64(-math.pi), wp.float64(math.pi)),
            wp.clamp(angles[1] + step[1], wp.float64(-math.pi), wp.float64(math.pi)),
        )
    return angles, False


@wp.func
def _release_sweep_floor(proximal: wp.vec2d, distal_angle: wp.float64, circle_radius: wp.float64, straight_count: int):
    orientation = proximal[0] + proximal[1]
    length = wp.float64(wp.static(LINK_LENGTHS[-1]))
    curved_count = wp.static(SECTION_COUNTS[2]) - straight_count
    entry_height = circle_radius * wp.sin(orientation - distal_angle / wp.float64(2.0))
    lowest = entry_height - wp.max(wp.float64(0.0), wp.float64(straight_count) * length * wp.cos(orientation))
    for sample in range(65):
        angle = distal_angle * wp.float64(sample) / wp.float64(64.0)
        height = entry_height
        for index in range(curved_count):
            height += length * wp.cos(orientation + wp.float64(index) * angle)
            lowest = wp.min(lowest, height)
    # Bound motion between samples, so the whole opening sweep clears the rim.
    sample_error = length * wp.float64(curved_count * (curved_count - 1)) * distal_angle / wp.float64(256.0)
    return lowest - sample_error


@wp.kernel
def _build_targets(
    object_center: wp.vec2d,
    target_center: wp.vec2d,
    target_yaw: wp.float64,
    distal_angle: wp.float64,
    circle_radius: wp.float64,
    straight_count: int,
    lift_height: wp.float64,
    cup_rim: wp.float64,
    surface_offset: wp.float64,
    targets: wp.array2d(dtype=wp.float64),
    failed: wp.array(dtype=int),
    failed_target: wp.array(dtype=wp.vec2d),
    release_height: wp.array(dtype=wp.float64),
):
    approach, converged = _solve_center(object_center, wp.vec2d(wp.float64(0.3)), distal_angle, circle_radius, straight_count)
    if not converged:
        failed[0] = 1
        failed_target[0] = object_center
        return
    release_center = target_center
    release_center[1] = cup_rim + circle_radius + surface_offset + wp.float64(0.02)
    release_guess = approach
    release_found = bool(False)
    for _ in range(80):
        release_guess, converged = _solve_center(release_center, release_guess, distal_angle, circle_radius, straight_count)
        if converged:
            floor = release_center[1] + _release_sweep_floor(release_guess, distal_angle, circle_radius, straight_count) - surface_offset
            if floor >= cup_rim + wp.float64(0.02):
                release_found = True
                break
        release_center[1] += wp.float64(0.005)
    if not release_found:
        failed[0] = 1
        failed_target[0] = release_center
        return
    release_height[0] = release_center[1]
    pregrasp_center = object_center + wp.vec2d(wp.float64(0.0), wp.float64(0.02))
    approach, converged = _solve_center(pregrasp_center, approach, distal_angle, circle_radius, straight_count)
    if not converged:
        failed[0] = 1
        failed_target[0] = pregrasp_center
        return
    previous = approach
    # Each solution seeds the next timestamp, preserving the motion history.
    for index in range(targets.shape[0]):
        timestamp = wp.float64(index) / wp.float64(60.0)
        motion_time = wp.max(timestamp - wp.float64(1.0), wp.float64(0.0))
        yaw = target_yaw * _smoothstep((motion_time - wp.float64(6.5)) / wp.float64(4.0))
        proximal = previous
        distal = wp.float64(0.0)
        # Curl above the scene before approaching the pickup cradle.
        if motion_time < wp.float64(2.0):
            proximal = approach * _smoothstep(motion_time / wp.float64(2.0))
            distal = wp.float64(0.20) * _smoothstep(timestamp)
        elif motion_time < wp.float64(4.0):
            fraction = _smoothstep((motion_time - wp.float64(2.0)) / wp.float64(2.0))
            center = pregrasp_center + (object_center - pregrasp_center) * fraction
            previous, converged = _solve_center(center, previous, distal_angle, circle_radius, straight_count)
            if not converged:
                failed[0] = 1
                failed_target[0] = center
                return
            proximal = previous
            distal = wp.float64(0.20) + (distal_angle - wp.float64(0.20)) * fraction
        elif motion_time < wp.float64(14.0):
            center = object_center
            center[1] += (lift_height - object_center[1]) * _smoothstep((motion_time - wp.float64(4.5)) / wp.float64(2.0))
            if motion_time >= wp.float64(10.5):
                center += (release_center - center) * _smoothstep((motion_time - wp.float64(10.5)) / wp.float64(3.0))
            previous, converged = _solve_center(center, previous, distal_angle, circle_radius, straight_count)
            if not converged:
                failed[0] = 1
                failed_target[0] = center
                return
            proximal = previous
            distal = distal_angle
        elif motion_time < wp.float64(16.0):
            distal = distal_angle * (wp.float64(1.0) - _smoothstep((motion_time - wp.float64(14.0)) / wp.float64(2.0)))
        else:
            proximal = previous * (wp.float64(1.0) - _smoothstep((motion_time - wp.float64(16.0)) / wp.float64(2.0)))
        targets[index, 0] = yaw
        for joint in range(wp.static(len(LINK_LENGTHS))):
            target = distal
            if joint < wp.static(SECTION_COUNTS[0]):
                target = proximal[0] / wp.float64(wp.static(SECTION_COUNTS[0]))
            elif joint < wp.static(sum(SECTION_COUNTS[:2])):
                target = proximal[1] / wp.float64(wp.static(SECTION_COUNTS[1]))
            elif joint <= wp.static(sum(SECTION_COUNTS[:2])) + straight_count:
                target = wp.float64(0.0)
            targets[index, joint + 1] = target


class GraspPlan:
    """Plan three PCC section bends to grasp, carry, release, and retreat.

    Equal internal joint increments discretize each section's curvature target.
    The distal entry stays tangent to the middle section to clear the closing tip.
    Newton's compliant dynamics can deform that target under load and contact.
    Base yaw places the three-section bending plane at the cup's azimuth.
    This bounded demonstration does not search for arbitrary grasps or
    predict whether frictional contact will retain an object.
    """

    duration = 21.0

    def __init__(self, grasp_bend=None, *, ball_radius=OBJECT_RADIUS, rail_height=0.235, cup_position=None, device=None):
        if not math.isfinite(ball_radius) or not 0 < ball_radius < CUP_INNER_RADIUS:
            raise ValueError("ball_radius must be positive and smaller than the cup's 0.055 m inner radius")
        if not math.isfinite(rail_height) or rail_height <= 0:
            raise ValueError("rail_height must be finite and positive")
        self.ball_radius = float(ball_radius)
        self.rail_height = float(rail_height)
        self.object_center = np.array([OBJECT_CENTER[0], OBJECT_CENTER[1], self.rail_height + 0.6 * self.ball_radius])
        distal_thicknesses = LINK_THICKNESSES[-SECTION_COUNTS[2]:]
        surface_offset = max(distal_thicknesses) / 2 + GRIP_RIDGE_HEIGHT
        self._straight_count = 0
        if grasp_bend is None:
            self._circle_radius = self.ball_radius + float(np.mean(distal_thicknesses)) / 2 + GRIP_RIDGE_HEIGHT
            self._distal_angle = 2 * math.asin(LINK_LENGTHS[-1] / (2 * self._circle_radius))
            tip_clearance = surface_offset + distal_thicknesses[-1] / 2 + 0.003
            max_arc = 2 * math.pi - 2 * math.asin(min(1.0, tip_clearance / (2 * self._circle_radius)))
            curved_count = min(SECTION_COUNTS[2], math.floor(max_arc / self._distal_angle))
            self._straight_count = SECTION_COUNTS[2] - curved_count
            grasp_bend = curved_count * self._distal_angle
            # A straight prefix keeps small-ball wraps from curling through their own entry.
            opening = 2 * self._circle_radius * math.sin(grasp_bend / 2)
            if opening - surface_offset - distal_thicknesses[-1] / 2 >= 2 * self.ball_radius:
                raise ValueError("ball_radius cannot be enclosed by the distal links without self-overlap")
        else:
            if not math.isfinite(grasp_bend) or not math.pi < grasp_bend < 2 * math.pi:
                raise ValueError("grasp_bend must lie between pi and 2*pi radians")
            self._distal_angle = grasp_bend / SECTION_COUNTS[2]
            self._circle_radius = LINK_LENGTHS[-1] / (2 * math.sin(self._distal_angle / 2))
        if not math.isfinite(grasp_bend) or not math.pi < grasp_bend < 2 * math.pi:
            raise ValueError("grasp_bend must lie between pi and 2*pi radians")
        self.grasp_bend = float(grasp_bend)
        self.cup_position = np.array([-0.27, 0.0, CUP_FLOOR_Z] if cup_position is None else cup_position, dtype=float)
        if self.cup_position.shape != (3,) or not np.isfinite(self.cup_position).all():
            raise ValueError("cup_position must contain three finite coordinates")
        radius = np.linalg.norm(self.cup_position[:2])
        if not CUP_RADIUS_RANGE[0] - 1e-12 <= radius <= CUP_RADIUS_RANGE[1] + 1e-12 or abs(self.cup_position[2] - CUP_FLOOR_Z) > 1e-9:
            raise ValueError("The cup must rest on the ground within the reachable annulus")
        if np.linalg.norm(self.cup_position[:2] - self.object_center[:2]) < CUP_SOURCE_CLEARANCE:
            raise ValueError("The cup overlaps the pickup fixture's clearance region")
        self.target_yaw = float(np.arctan2(self.cup_position[1], self.cup_position[0]))
        rim = self.cup_position[2] + CUP_FLOOR_THICKNESS / 2 + CUP_WALL_HEIGHT
        # Lift above a conservative gripper envelope before rotating over the scene.
        sweep_height = rim + self._circle_radius + sum(LINK_LENGTHS[-SECTION_COUNTS[2]:]) + surface_offset + 0.012
        self.target_center = np.array([self.cup_position[0], self.cup_position[1], sweep_height])
        self.lift_height = max(self.rail_height + self._circle_radius + surface_offset + 0.025,
                               sweep_height + self.ball_radius)
        self._targets = wp.empty((int(self.duration * 60) + 1, len(LINK_LENGTHS) + 1), dtype=wp.float64, device=device)
        failed = wp.zeros(1, dtype=int, device=self._targets.device)
        failed_target = wp.empty(1, dtype=wp.vec2d, device=self._targets.device)
        release_height = wp.empty(1, dtype=wp.float64, device=self._targets.device)
        wp.launch(
            _build_targets, dim=1,
            inputs=[wp.vec2d(self.object_center[0], self.object_center[2]), wp.vec2d(radius, self.target_center[2]),
                    self.target_yaw, self._distal_angle, self._circle_radius, self._straight_count, self.lift_height,
                    rim, surface_offset],
            outputs=[self._targets, failed, failed_target, release_height], device=self._targets.device,
        )
        if failed.numpy()[0]:
            raise ValueError(f"The grasp center {failed_target.numpy()[0].tolist()} is outside the planner's reachable path")
        self.target_center[2] = release_height.numpy()[0]
