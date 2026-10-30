"""Run live simulations with an optional Newton desktop viewer."""

import newton.viewer
import warp as wp

def render(viewer, simulation):
    viewer.begin_frame(simulation.time)
    viewer.log_state(simulation.state)
    viewer.log_contacts(simulation.contacts, simulation.state)
    viewer.end_frame()


def run_simulation(
    simulation,
    *,
    headless=True,
    camera_position=(0.95, 0.5, 1.1),
    camera_pitch=-30.0,
    camera_yaw=-135.0,
    draw_ui=None,
    world_spacing=None,
):
    if headless:
        while True:
            simulation.step()

    viewer = newton.viewer.ViewerGL(width=2000, height=800)
    try:
        viewer.set_model(simulation.model)
        if world_spacing is not None:
            viewer.set_world_offsets(world_spacing)
        viewer.set_reset_callback(simulation.reset)
        # viewer.show_contacts = True
        viewer.set_camera(wp.vec3(*camera_position), pitch=camera_pitch, yaw=camera_yaw)
        if draw_ui is not None:
            viewer.register_ui_callback(lambda imgui: draw_ui(imgui, viewer, simulation), position="free")
        
        while viewer.is_running():
            if not viewer.should_step():
                render(viewer, simulation)
                continue
            simulation.step()
            render(viewer, simulation)
        
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()
