"""Choose a Newton arm experiment; existing commands still run the baseline."""

import argparse
import sys


DEFAULT_SIM_PARAMS = {"substeps": 4, "iterations": 10}
BASELINE_CONTROL_PARAMS = {"gain": 4e4, "damping": 1.2e3, "ramp_time": 2.5}
DEFAULT_FRAME_DT = 1.0 / 120
DEFAULT_BASELINE_INPUTS = [5.03, -2.65, 6.59, 1.79]


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in ("-h", "--help"):
        parser = argparse.ArgumentParser(
            description=__doc__,
            epilog="Use 'continuum-robot baseline --help' or 'continuum-robot grasping --help' for simulation options.",
        )
        parser.add_argument("scenario", nargs="?", choices=("baseline", "grasping"), default="baseline")
        parser.print_help()
        return
    scenario = args.pop(0) if args and args[0] in ("baseline", "grasping") else "baseline"
    if scenario == "grasping":
        from .grasping.cli import main as run
    else:
        from .baseline.cli import main as run
    return run(args)
