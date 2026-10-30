"""PCC centerline target forces with velocity feedforward."""

import warp as wp

@wp.kernel
def update_targets(
    curr_time: wp.array[wp.float64],
    time_offset: wp.float64,
    samples: wp.array3d[wp.float32],
    targets: wp.array[wp.vec3],
    target_velocities: wp.array[wp.vec3],
):
    world, segment = wp.tid()
    sample = wp.clamp((curr_time[0] + time_offset) * wp.float64(60.0), wp.float64(0.0), wp.float64(samples.shape[1] - 1))
    lower = int(wp.floor(sample))
    upper = wp.min(lower + 1, samples.shape[1] - 1)
    alpha = float(sample - wp.float64(lower))
    point = wp.vec3(0.0)
    velocity = wp.vec3(0.0)
    for axis in range(3):
        column = segment * 3 + axis
        delta = samples[world, upper, column] - samples[world, lower, column]
        point[axis] = samples[world, lower, column] + alpha * delta
        velocity[axis] = 60.0 * delta
    targets[world * (samples.shape[2] // 3) + segment] = point
    target_velocities[world * (samples.shape[2] // 3) + segment] = velocity


@wp.kernel
def pcc_forces(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_inv_mass: wp.array[float],
    rod_bodies: wp.array[int],
    targets: wp.array[wp.vec3],
    target_velocities: wp.array[wp.vec3],
    gain: float,
    damping: float,
    dt: float,
    body_f: wp.array[wp.spatial_vector],
):
    segment = wp.tid()
    body = rod_bodies[segment]
    if body_inv_mass[body] > 0.0:
        velocity = wp.spatial_top(body_qd[body])
        error = targets[segment] - wp.transform_get_translation(body_q[body])
        # Implicit spring/damper evaluated at the next substep's moving target.
        kd = damping + gain * dt
        force = (gain * error + damping * target_velocities[segment] - kd * velocity) / (1.0 + kd * dt * body_inv_mass[body])
        body_f[body] = wp.spatial_vector(force, wp.vec3(0.0))
