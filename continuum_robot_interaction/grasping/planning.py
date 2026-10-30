"""A staged PCC grasp and cup deposit with a rotating bending plane.

The previous IK solution seeds the next one, following the motion-history idea
in OpenTDCRContactModel (https://github.com/ContinuumRoboticsLab/OpenTDCRContactModel).
This is geometric planning; Newton resolves the resulting object contacts.
"""

import math

import numpy as np
import warp as wp

from .model import BASE_HEIGHT, SEGMENT_LENGTHS, SEGMENT_DIAMETERS, SECTION_COUNTS


OBJECT_CENTER = np.array([0.17, 0.0, 0.25])
OBJECT_RADIUS = 0.025
CUP_RADIUS_RANGE = (0.18, 0.29)
CUP_SOURCE_CLEARANCE = 0.100
CUP_FLOOR_Z = 0.004
CUP_INNER_RADIUS = 0.055
CUP_WALL_HEIGHT = 0.06
CUP_FLOOR_THICKNESS = 0.008
# Separate stationary release holds so shorter pauses preserve the opening speed.
# Open more slowly while unloading contact, then accelerate the free tip.
PATH_TIMES = (0., 1., 3., 5., 5.5, 7.5, 15.5, 18.5, 21., 22., 23., 24., 25., 26., 28., 30.)
MOTION_TIMES = (0., .25, 1., 1.75, 2., 2.5, 5., 5.5, 6., 6.3, 6.4, 6.8, 7., 7.1, 7.35, 7.45)
SMALL_WRAP_MOTION_TIMES = (0., .25, .75, 1.35, 1.5, 2., 5., 5.5, 6., 6.3, 6.4, 6.8, 7., 7.1, 7.35, 7.45)

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
        length = wp.float64(wp.static(SEGMENT_LENGTHS[0]))
        if index >= wp.static(SECTION_COUNTS[0]):
            angle = proximal[1] / wp.float64(wp.static(SECTION_COUNTS[1]))
            length = wp.float64(wp.static(SEGMENT_LENGTHS[SECTION_COUNTS[0]]))
        orientation += angle
        center += length * wp.vec2d(wp.sin(orientation), wp.cos(orientation))
    center += wp.float64(straight_count) * wp.float64(wp.static(SEGMENT_LENGTHS[-1])) * wp.vec2d(wp.sin(orientation), wp.cos(orientation))
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
    length = wp.float64(wp.static(SEGMENT_LENGTHS[-1]))
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
    # Centered opening clears the cup below the fixed-backbone sweep bound.
    release_center[1] -= wp.float64(0.04)
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
        yaw = target_yaw * _smoothstep((motion_time - wp.float64(6.5)) / wp.float64(8.0))
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
        elif motion_time < wp.float64(22.0):
            center = object_center
            center[1] += (lift_height - object_center[1]) * _smoothstep((motion_time - wp.float64(4.5)) / wp.float64(2.0))
            if motion_time >= wp.float64(14.5):
                center += (release_center - center) * _smoothstep((motion_time - wp.float64(14.5)) / wp.float64(3.0))
            previous, converged = _solve_center(center, previous, distal_angle, circle_radius, straight_count)
            if not converged:
                failed[0] = 1
                failed_target[0] = center
                return
            proximal = previous
            distal = distal_angle
        elif motion_time < wp.float64(25.0):
            fraction = _smoothstep((motion_time - wp.float64(22.0)) / wp.float64(2.0))
            distal = distal_angle + (wp.float64(0.25) - distal_angle) * fraction
            # Expand about the ball instead of sweeping the closing tip across it.
            radius = wp.float64(wp.static(SEGMENT_LENGTHS[-1])) / (wp.float64(2.0) * wp.sin(distal / wp.float64(2.0)))
            previous, converged = _solve_center(release_center, previous, distal, radius, straight_count)
            if not converged:
                failed[0] = 1
                failed_target[0] = release_center
                return
            proximal = previous
        else:
            fraction = wp.float64(1.0) - _smoothstep((motion_time - wp.float64(25.0)) / wp.float64(2.0))
            proximal = previous * fraction
            distal = wp.float64(0.25) * fraction
        targets[index, 0] = yaw
        for joint in range(wp.static(len(SEGMENT_LENGTHS))):
            target = distal
            if joint < wp.static(SECTION_COUNTS[0]):
                target = proximal[0] / wp.float64(wp.static(SECTION_COUNTS[0]))
            elif joint < wp.static(sum(SECTION_COUNTS[:2])):
                target = proximal[1] / wp.float64(wp.static(SECTION_COUNTS[1]))
            elif joint <= wp.static(sum(SECTION_COUNTS[:2])) + straight_count:
                target = wp.float64(0.0)
            targets[index, joint + 1] = target


def _cup_nodes(nodes, start_node, radius, nominal_times):
    """Cup the closed wrap during carry, preserving its anchor and chord lengths."""
    result = np.asarray(nodes, dtype=np.float64).copy()
    times = np.asarray(nominal_times)
    close = np.clip((times - 6.0) / 1.5, 0.0, 1.0)
    release = np.clip((times - 19.0) / 2.0, 0.0, 1.0)
    blend = close**2 * (3 - 2 * close) * (1 - release**2 * (3 - 2 * release))
    selected = blend > 0.0
    if not selected.any():
        return result
    original = result[selected, start_node:].copy()
    edges = np.diff(original, axis=1)
    lengths = np.linalg.norm(edges, axis=2)
    first = edges[:, 0] / lengths[:, 0, None]
    normal = np.cross(first, edges[:, 1])
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    center = (original[:, 0] + lengths[:, 0, None] * first / 2
              + np.sqrt(radius**2 - lengths[:, 0, None]**2 / 4) * np.cross(normal, first))
    u = (original[:, 0] - center) / radius
    v = np.cross(normal, u)
    scale = 0.7 * blend[selected]

    def point(angle):
        latitude = scale * (1 - np.cos(angle)) * np.sin(angle)
        return center + radius * (np.cos(latitude)[:, None]
            * (u * np.cos(angle)[:, None] + v * np.sin(angle)[:, None])
            + normal * np.sin(latitude)[:, None])

    formed = original.copy()
    theta = np.zeros(selected.sum())
    for segment in range(lengths.shape[1]):
        length = lengths[:, segment]
        delta = 2 * np.arcsin(length / (2 * radius))
        low, high = theta.copy(), theta + 1.5 * delta
        # Expand the search interval where the spherical curve runs along a latitude.
        short = np.linalg.norm(point(high) - formed[:, segment], axis=1) < length
        high = np.where(short, theta + 2 * delta, high)
        if np.any(np.linalg.norm(point(high) - formed[:, segment], axis=1) < length):
            raise ValueError("Cupping amplitude cannot bracket the next rod segment")
        for _ in range(30):
            midpoint = (low + high) / 2
            too_long = np.linalg.norm(point(midpoint) - formed[:, segment], axis=1) > length
            high = np.where(too_long, midpoint, high)
            low = np.where(too_long, low, midpoint)
        theta = (low + high) / 2
        formed[:, segment + 1] = point(theta)
    assert np.allclose(np.linalg.norm(np.diff(formed, axis=1), axis=2), lengths, rtol=0, atol=1e-9)
    result[selected, start_node:] = formed
    return result


def _centerline_nodes(samples, target_center, circle_radius, straight_count):
    """Roll the held wrap over the cup without moving its center or stretching rods."""
    headings = np.cumsum(samples[:, 1:], axis=1)
    yaw = samples[:, 0]
    radial = np.column_stack((np.cos(yaw), np.sin(yaw), np.zeros_like(yaw)))
    tangents = np.sin(headings)[:, :, None] * radial[:, None, :]
    tangents[:, :, 2] = np.cos(headings)
    increments = tangents * np.asarray(SEGMENT_LENGTHS)[None, :, None]
    nodes = np.concatenate((np.zeros((len(samples), 1, 3)), np.cumsum(increments, axis=1)), axis=1)
    base = np.array([0.0, 0.0, BASE_HEIGHT])
    nodes += base
    times = np.arange(len(samples)) / 60.0
    nodes = _cup_nodes(nodes, sum(SECTION_COUNTS[:2]) + straight_count, circle_radius, times)
    axis = np.asarray(target_center) - base
    axis = axis / np.linalg.norm(axis)
    # Finish tipping before opening so the ball can fall without a sideways sweep.
    close = np.clip((times - 19.0) / 3.0, 0.0, 1.0)
    retreat = np.clip((times - 26.0) / 2.0, 0.0, 1.0)
    roll = (math.pi / 2) * close**2 * (3 - 2 * close) * (1 - retreat**2 * (3 - 2 * retreat))
    offsets = nodes - base
    cosine, sine = np.cos(roll)[:, None, None], np.sin(roll)[:, None, None]
    # A fixed base-to-grasp axis preserves both anchors throughout opening.
    return (base + offsets * cosine + np.cross(axis, offsets) * sine
            + axis * np.sum(offsets * axis, axis=2)[:, :, None] * (1 - cosine))


def _closed_wrap(ball_radius, thicknesses, grasp_bend=None):
    """Keep preload while clearing both the distal entry and the middle rod."""
    length = SEGMENT_LENGTHS[-1]
    count = SECTION_COUNTS[-1]
    rod_radius = max(thicknesses) / 2
    middle_radius = SEGMENT_DIAMETERS[-count - 1] / 2
    # Balance coverage and preload where the number of curved segments changes.
    nominal_radius = 0.92 * ball_radius + float(np.mean(thicknesses)) / 2

    def clearance(radius, curved_count):
        angle = 2 * math.asin(length / (2 * radius))
        x = radius * (math.cos(angle / 2) - math.cos((curved_count - 0.5) * angle))
        z = radius * (math.sin(angle / 2) + math.sin((curved_count - 0.5) * angle))
        prefix = (count - curved_count) * length
        middle_gap = math.hypot(x, max(0.0, z + prefix)) - middle_radius - rod_radius - 0.003
        if prefix == 0.0:
            return middle_gap
        entry_gap = math.hypot(x, max(0.0, z, -prefix - z)) - 2 * rod_radius - 0.003
        return min(middle_gap, entry_gap)

    if grasp_bend is not None:
        angle = grasp_bend / count
        radius = length / (2 * math.sin(angle / 2))
        if clearance(radius, count) < 0.0:
            raise ValueError("grasp_bend makes the tip collide with the entry rod")
        return radius, angle, count

    angle = 2 * math.asin(length / (2 * nominal_radius))
    for curved_count in range(count, 2, -1):
        if math.pi < curved_count * angle < 2 * math.pi and clearance(nominal_radius, curved_count) >= 0.0:
            break
    else:
        raise ValueError("ball_radius cannot be enclosed without the tip colliding with the entry rod")
    radius = nominal_radius
    if curved_count < count:
        low = length / (2 * math.sin(math.pi / curved_count))
        high = nominal_radius
        for _ in range(35):
            midpoint = (low + high) / 2
            if clearance(midpoint, curved_count) < 0.0:
                low = midpoint
            else:
                high = midpoint
        # Recover some closure lost to whole-segment rounding, leaving room for deformation.
        radius = (nominal_radius + 3 * high) / 4
    return radius, 2 * math.asin(length / (2 * radius)), curved_count


class GraspPlan:
    """Plan three PCC section bends to grasp, carry, release, and retreat.

    Equal angular increments discretize each section's curvature target.
    The distal entry stays tangent to the middle section to clear the closing tip.
    These geometric samples are controller references, not rod joint drives.
    Yaw places the three-section bending plane at the cup's azimuth.
    Over the cup, a spatial roll tips the held wrap before a centered opening.
    This bounded demonstration does not search for arbitrary grasps or
    predict whether frictional contact will retain an object.
    """

    duration = MOTION_TIMES[-1]

    def __init__(self, grasp_bend=None, *, ball_radius=OBJECT_RADIUS, rail_height=0.235, cup_position=None, device=None):
        if not math.isfinite(ball_radius) or not 0 < ball_radius < CUP_INNER_RADIUS:
            raise ValueError("ball_radius must be positive and smaller than the cup's 0.055 m inner radius")
        if not math.isfinite(rail_height) or rail_height <= 0:
            raise ValueError("rail_height must be finite and positive")
        self.ball_radius = float(ball_radius)
        self.rail_height = float(rail_height)
        self.object_center = np.array([OBJECT_CENTER[0], OBJECT_CENTER[1], self.rail_height + 0.6 * self.ball_radius])
        distal_thicknesses = SEGMENT_DIAMETERS[-SECTION_COUNTS[2]:]
        surface_offset = max(distal_thicknesses) / 2
        self._straight_count = 0
        if grasp_bend is None:
            self._circle_radius, self._distal_angle, curved_count = _closed_wrap(self.ball_radius, distal_thicknesses)
            self._straight_count = SECTION_COUNTS[2] - curved_count
            grasp_bend = curved_count * self._distal_angle
            # A straight prefix keeps small-ball wraps from curling through their own entry.
            opening = 2 * self._circle_radius * math.sin(grasp_bend / 2)
            if opening - surface_offset - distal_thicknesses[-1] / 2 >= 2 * self.ball_radius:
                raise ValueError("ball_radius cannot be enclosed by the distal rod segments without self-overlap")
        else:
            if not math.isfinite(grasp_bend) or not math.pi < grasp_bend < 2 * math.pi:
                raise ValueError("grasp_bend must lie between pi and 2*pi radians")
            self._circle_radius, self._distal_angle, _ = _closed_wrap(self.ball_radius, distal_thicknesses, grasp_bend)
        if not math.isfinite(grasp_bend) or not math.pi < grasp_bend < 2 * math.pi:
            raise ValueError("grasp_bend must lie between pi and 2*pi radians")
        self.grasp_bend = float(grasp_bend)
        # Small wraps need a slower turn; their quicker pickup keeps the cycle bounded.
        self.motion_times = SMALL_WRAP_MOTION_TIMES if self._straight_count >= 4 else MOTION_TIMES
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
        sweep_height = rim + self._circle_radius + sum(SEGMENT_LENGTHS[-SECTION_COUNTS[2]:]) + surface_offset + 0.012
        self.target_center = np.array([self.cup_position[0], self.cup_position[1], sweep_height])
        self.lift_height = max(self.rail_height + self._circle_radius + surface_offset + 0.025,
                               sweep_height + self.ball_radius)
        self._targets = wp.empty((int(PATH_TIMES[-1] * 60) + 1, len(SEGMENT_LENGTHS) + 1), dtype=wp.float64, device=device)
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
        nodes = _centerline_nodes(self._targets.numpy(), self.target_center, self._circle_radius, self._straight_count)
        midpoints = (nodes[:, :-1] + nodes[:, 1:]) / 2
        path = midpoints.reshape(len(nodes), -1)
        times = np.arange(round(self.duration * 60) + 1) / 60.0
        indices = np.interp(times, self.motion_times, PATH_TIMES) * 60.0
        lower = np.floor(indices).astype(int)
        upper = np.minimum(lower + 1, len(path) - 1)
        alpha = (indices - lower)[:, None]
        self._targets = wp.array(path[lower] * (1 - alpha) + path[upper] * alpha,
                                 dtype=wp.float32, device=self._targets.device)
