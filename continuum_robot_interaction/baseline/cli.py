"""Run the baseline two-section arm and fixed-cylinder interaction with Newton."""

import argparse

from ..cli import (
    BASELINE_CONTROL_PARAMS,
    DEFAULT_BASELINE_INPUTS,
    DEFAULT_SIM_PARAMS,
)
from . import sim


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs", type=float, nargs=4, default=DEFAULT_BASELINE_INPUTS,
        metavar=("K1", "PHI1", "K2", "PHI2"),
        help="section curvatures (1/m) and bending angles (radians)",
    )
    parser.add_argument("--device", default=None, help="Warp device, e.g. cpu or cuda:0")
    parser.add_argument("--headless", action="store_true", help="simulate without a viewer")
    parser.add_argument("--gain", type=float, default=BASELINE_CONTROL_PARAMS["gain"], help="PCC controller spring gain")
    parser.add_argument("--damping", type=float, default=BASELINE_CONTROL_PARAMS["damping"], help="PCC controller damping")
    parser.add_argument("--substeps", type=int, default=DEFAULT_SIM_PARAMS["substeps"], help="physics steps per simulation frame (must be even)")
    parser.add_argument("--iterations", type=int, default=DEFAULT_SIM_PARAMS["iterations"], help="Newton solver iterations per physics step")
    parser.add_argument("--ramp-time", type=float, default=BASELINE_CONTROL_PARAMS["ramp_time"], help="seconds to smoothly reach the requested curvatures (0 for an immediate command)")
    args = parser.parse_args(argv)

    sim.run(
        args.inputs,
        params={
            "gain": args.gain, "damping": args.damping,
            "substeps": args.substeps, "iterations": args.iterations, "ramp_time": args.ramp_time,
        },
        device=args.device,
        headless=args.headless,
    )
