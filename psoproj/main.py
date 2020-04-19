
# run from {repo_root} with: python ./psoproj/main.py --help
import argparse
import sys

from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Type

import numpy as np
import pyswarms
from pyswarms.base import SwarmOptimizer
from pyswarms.utils.functions import single_obj

from .optimizers import get_optimizer_map, mutagenic_pso
from .util import is_objective_func, is_optimizer


def get_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-f", "--function",
        dest="objective_function",
        action="store",
        type=is_objective_func,
        default="rastrigin",
        help="the objective function to optimize. default: %(default)s"
    )
    ap.add_argument(
        "-m" "--optimization-method",
        dest="optimizer",
        action="store",
        type=is_optimizer,
        choices=get_optimizer_map().keys(),
        default="global_best",
        help="the swarm optimizer to use. default: %(default)s"
    )
    return ap

def main():
    args = get_argparser().parse_args()

    optimizer_class = get_optimizer_map()[args.optimizer]
    objective_function = getattr(single_obj, args.objective_function)

    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
