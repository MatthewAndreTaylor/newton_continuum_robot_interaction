"""Scene-anchored dimension lines for the grasping viewer."""

import numpy as np


def draw_measurements(imgui, viewer, simulation):
    camera = viewer.camera
    projection = camera.get_projection_matrix().reshape(4, 4).T @ camera.get_view_matrix().reshape(4, 4).T
    width, height = imgui.get_io().display_size
    draw = imgui.get_background_draw_list()
    scale = imgui.get_font_size() / 16.0

    def project(point):
        clip = projection @ np.append(point, 1.0)
        if clip[3] <= 0.0:
            return None
        ndc = clip[:3] / clip[3]
        if not -1.0 <= ndc[2] <= 1.0:
            return None
        return np.array([(ndc[0] + 1.0) * width / 2, (1.0 - ndc[1]) * height / 2])

    def dimension(start, end, label, color, outward):
        a, b = project(start), project(end)
        if a is None or b is None:
            return
        delta = b - a
        length = np.linalg.norm(delta)
        if length < 2.0:
            return
        normal = np.array([delta[1], -delta[0]]) / length
        if np.dot(normal, outward) < 0.0:
            normal = -normal
        offset = normal * 28.0 * scale
        first, last = a + offset, b + offset
        draw.add_line(tuple(first), tuple(last), color, 2.0 * scale)
        for anchor, tip in ((a, first), (b, last)):
            draw.add_line(tuple(anchor), tuple(tip + normal * 5.0 * scale), color, scale)
            tick = normal * 5.0 * scale
            draw.add_line(tuple(tip - tick), tuple(tip + tick), color, 2.0 * scale)
        text_size = np.array(tuple(imgui.calc_text_size(label)))
        center = (first + last) / 2 + normal * (np.dot(np.abs(normal), text_size) / 2 + 8.0 * scale)
        origin = center - text_size / 2
        padding = 4.0 * scale
        background = imgui.get_color_u32((0.04, 0.06, 0.09, 0.9))
        draw.add_rect_filled(tuple(origin - padding), tuple(origin + text_size + padding), background, 4.0 * scale)
        draw.add_text(tuple(origin), color, label)

    right = np.array(camera.get_right())
    rail_color = imgui.get_color_u32((0.45, 0.85, 1.0, 1.0))
    base = np.array([simulation.object_initial_position[0], simulation.rail_offset, 0.0])
    top = base + [0.0, 0.0, simulation.rail_height]
    dimension(base, top, f"height: {simulation.rail_height * 1000:.0f} mm", rail_color, (1.0, 0.0))

    center = simulation.state.body_q.numpy()[simulation.object_body, :3]
    radius = simulation.object_radius
    ball_color = imgui.get_color_u32((1.0, 0.75, 0.3, 1.0))
    dimension(center - right * radius, center + right * radius,
              f"diameter: {2 * radius * 1000:.0f} mm", ball_color, (0.0, -1.0))
