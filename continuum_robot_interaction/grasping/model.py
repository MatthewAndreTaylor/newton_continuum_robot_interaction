"""Dimensions of the tapered, three-section bending arm, in metres.

The URDF is a rigid-link discretization: Newton joint drives supply compliant
bending. The continuous base yaw rotates about +Z; each bending joint rotates
about local +Y at the proximal end of its box link. The straight arm points +Z.
"""

from pathlib import Path


URDF_PATH = Path(__file__).with_name("assets") / "three_section_arm.urdf"
CUP_URDF_PATH = URDF_PATH.with_name("cup.urdf")
BASE_HEIGHT = 0.04
# Ridges extend from the distal core boxes' local +X faces; lateral widths stay unchanged.
GRIP_RIDGE_WIDTH = 0.002
GRIP_RIDGE_HEIGHT = 0.003
SECTION_LENGTHS = (0.16, 0.28, 0.19)
SECTION_COUNTS = (6, 6, 12)
SECTION_SLICES = (slice(0, 6), slice(6, 12), slice(12, 24))
SECTION_WIDTHS = (0.0775, 0.0405, 0.013)
SECTION_THICKNESSES = (0.041, 0.024, 0.0125)
LINK_LENGTHS = tuple(
    length / count
    for length, count in zip(SECTION_LENGTHS, SECTION_COUNTS)
    for _ in range(count)
)
LINK_WIDTHS = tuple(
    start + (end - start) * (index + 0.5) / count
    for start, end, count in zip((0.09, 0.065, 0.016), (0.065, 0.016, 0.01), SECTION_COUNTS)
    for index in range(count)
)
LINK_THICKNESSES = tuple(
    start + (end - start) * (index + 0.5) / count
    for start, end, count in zip((0.05, 0.032, 0.016), (0.032, 0.016, 0.009), SECTION_COUNTS)
    for index in range(count)
)
JOINT_NAMES = tuple(
    f"section{section}_joint_{index:02d}"
    for section, count in enumerate(SECTION_COUNTS, start=1)
    for index in range(count)
)
LINK_NAMES = tuple(name.replace("joint", "link") for name in JOINT_NAMES)
YAW_JOINT_NAME = "base_yaw_joint"
ACTUATED_JOINT_NAMES = (YAW_JOINT_NAME, *JOINT_NAMES)
