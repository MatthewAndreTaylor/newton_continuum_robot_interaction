"""Three passive elastic rod sections for PCC target-force control, in SI units."""

from pathlib import Path

import newton
import numpy as np
import warp as wp


CUP_URDF_PATH = Path(__file__).resolve().parents[1] / "assets" / "cup.urdf"
BASE_HEIGHT = 0.04
SECTION_LENGTHS = (0.16, 0.28, 0.19)
SECTION_COUNTS = (6, 6, 12)
SECTION_RADII = (0.0105,) * len(SECTION_LENGTHS)
ROD_FRICTION = 5.0
# Balance backbone stiffness with the reduced cross-sectional area and mass.
SECTION_BEND_STIFFNESS = (2.0 * (SECTION_RADII[0] / 0.018)**2, (SECTION_RADII[1] / 0.012)**2, 0.008)
# The tip bends easily around objects but resists twisting during transport.
SECTION_TWIST_STIFFNESS = (5 * SECTION_BEND_STIFFNESS[0], 5 * SECTION_BEND_STIFFNESS[1], 0.32)
SEGMENT_LENGTHS = tuple(length / count for length, count in zip(SECTION_LENGTHS, SECTION_COUNTS) for _ in range(count))
SEGMENT_DIAMETERS = tuple(2 * radius for radius, count in zip(SECTION_RADII, SECTION_COUNTS) for _ in range(count))


def add_arm(builder):
    """Add passive rods attached to a fixed pedestal."""
    base = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, BASE_HEIGHT / 2), wp.quat_identity()), label="base")
    builder.add_shape_cylinder(base, radius=0.035, half_height=BASE_HEIGHT / 2,
                               cfg=newton.ModelBuilder.ShapeConfig(density=0.0, gap=0.001), color=(0.3, 0.3, 0.35))
    bodies, joints = [], []
    height = BASE_HEIGHT
    previous, previous_half_length = base, BASE_HEIGHT / 2
    colors = ((0.12, 0.45, 0.85), (0.95, 0.45, 0.12), (0.2, 0.7, 0.4))
    for section, (length, count, radius, stiffness, twist_stiffness) in enumerate(zip(
            SECTION_LENGTHS, SECTION_COUNTS, SECTION_RADII, SECTION_BEND_STIFFNESS, SECTION_TWIST_STIFFNESS)):
        segment_length = length / count
        points = np.zeros((count + 1, 3), dtype=np.float32)
        points[:, 2] = height + np.linspace(0.0, length, count + 1)
        rod = newton.Rod(points, radius=radius, stretch_rigidity=2e4 * segment_length,
                         shear_rigidity=2e4 * segment_length, bend_rigidity=stiffness * segment_length,
                         twist_rigidity=twist_stiffness * segment_length)
        grip_section = section == len(SECTION_LENGTHS) - 1
        damping = (0.2 if grip_section else 0.02) * stiffness
        section_bodies, section_joints = builder.add_rod(
            rod=rod, cfg=newton.ModelBuilder.ShapeConfig(density=700.0, mu=ROD_FRICTION,
                                                       ke=2e5, kd=100.0, gap=0.001),
            body_frame_origin="com", wrap_in_articulation=False, label=f"section_{section + 1}",
            bend_damping=damping, twist_damping=damping,
            stretch_damping=10.0, shear_damping=10.0, color=colors[section])
        root_joint = builder.add_joint_rod(
            previous, section_bodies[0],
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, previous_half_length), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, -segment_length / 2), wp.quat_identity()),
            stretch_stiffness=2e4, shear_stiffness=2e4, stretch_damping=10.0, shear_damping=10.0,
            bend_stiffness=stiffness, twist_stiffness=twist_stiffness,
            bend_damping=damping, twist_damping=damping, label=f"section_{section + 1}_root")
        joints.extend([*section_joints, root_joint])
        bodies.extend(section_bodies)
        previous, previous_half_length = section_bodies[-1], segment_length / 2
        height += length
    builder.add_articulation(joints, label="elastic_arm")
    # Thick capsules overlap nearby discretization neighbors even in a straight rod.
    shapes = {body: shape for shape, body in enumerate(builder.shape_body) if body in bodies}
    arc_lengths = np.concatenate(([0.0], np.cumsum(SEGMENT_LENGTHS)))
    for first, body in enumerate(bodies):
        for second in range(first + 2, len(bodies)):
            shape_a, shape_b = shapes[body], shapes[bodies[second]]
            gap = arc_lengths[second] - arc_lengths[first + 1]
            clearance = (SEGMENT_DIAMETERS[first] + SEGMENT_DIAMETERS[second]) / 2
            if gap <= clearance + builder.shape_gap[shape_a] + builder.shape_gap[shape_b]:
                builder.add_shape_collision_filter_pair(shape_a, shape_b)
    return base, bodies
