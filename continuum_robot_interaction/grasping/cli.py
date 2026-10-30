"""Grasp a ball using elastic rods and PCC target-force control."""

import argparse

from ..cli import DEFAULT_SIM_PARAMS
from . import sim


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=None, help="Warp device, e.g. cpu or cuda:0")
    parser.add_argument("--headless", action="store_true", help="simulate without a viewer")
    parser.add_argument("--num-envs", type=int, default=1, help="independent environments in one simulation; 576 displays a 24 x 24 grid")
    parser.add_argument("--seed", type=int, default=None, help="repeat the ball size and color, rail height, and cup placement; omitted chooses a new random seed")
    parser.add_argument("--ball-radius-range", type=float, nargs=2, metavar=("MIN", "MAX"), default=sim.BALL_RADIUS_RANGE, help=f"ball radius range in metres (default: {sim.BALL_RADIUS_RANGE}); equal endpoints fix the radius")
    parser.add_argument("--rail-height-range", type=float, nargs=2, metavar=("MIN", "MAX"), default=sim.RAIL_HEIGHT_RANGE, help=f"shared rail height range in metres (default: {sim.RAIL_HEIGHT_RANGE}); equal endpoints fix the height")
    parser.add_argument("--cup-position", type=float, nargs=2, metavar=("X", "Y"), help="place the cup at this ground position in metres, within arm reach")
    parser.add_argument("--gain", type=float, default=sim.GRASP_CONTROL_PARAMS["gain"], help="PCC controller spring gain")
    parser.add_argument("--damping", type=float, default=sim.GRASP_CONTROL_PARAMS["damping"], help="PCC controller damping")
    parser.add_argument("--substeps", type=int, default=DEFAULT_SIM_PARAMS["substeps"], help="physics steps per simulation frame (positive and even for graph capture)")
    parser.add_argument("--iterations", type=int, default=DEFAULT_SIM_PARAMS["iterations"], help="Newton solver iterations per physics step")
    args = parser.parse_args(argv)
    if args.num_envs < 1:
        parser.error("--num-envs must be positive")

    params = {"substeps": args.substeps, "iterations": args.iterations, "gain": args.gain, "damping": args.damping}
    run = sim.run
    batch_options = {}
    if args.num_envs > 1:
        from .batch import run

        batch_options["count"] = args.num_envs
    results = run(
        params=params,
        device=args.device,
        headless=args.headless,
        seed=args.seed,
        cup_position=(args.cup_position + [sim.CUP_GEOMETRY["floor_thickness"] / 2]
                      if args.cup_position is not None else None),
        ball_radius_range=args.ball_radius_range,
        rail_height_range=args.rail_height_range,
        **batch_options,
    )
    if args.num_envs > 1:
        deposited = sum(result["ball_in_cup"] for result in results)
        print(f"Environments: {len(results)}; seeds: {results[0]['seed']} through {results[-1]['seed']}")
        print(f"Balls in cups: {deposited}/{len(results)}")
        return
    cup = results["cup_position"]
    print(f"Seed: {results['seed']}; cup position: ({cup[0]:.3f}, {cup[1]:.3f}, {cup[2]:.3f}) m")
    print(f"Ball radius: {results['ball_radius']:.4f} m; diameter: {2 * results['ball_radius']:.4f} m; rail height: {results['rail_height']:.4f} m")
    print(f"Maximum object lift: {results['maximum_lift_height']:.3f} m; horizontal motion: {results['transport_distance']:.3f} m")
    print(f"Ball in cup: {results['ball_in_cup']}")
