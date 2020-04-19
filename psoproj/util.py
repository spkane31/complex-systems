import argparse

from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Type

import numpy as np
import pyswarms
from pyswarms.base import SwarmOptimizer
from pyswarms.utils.functions import single_obj

from .optimizers import mutagenic_pso

def get_optimizer_map() -> Dict[str, Type[SwarmOptimizer]]:
    opts = {
        "local_best": pyswarms.local_best.LocalBestPSO,
        "global_best": pyswarms.global_best.GlobalBestPSO,
        "mutagenic": mutagenic_pso.MutagenicPSO
    }
    return opts

def get_pyswarms_obj_funcs() -> Iterable[str]:
    funcs = {
        func_name
        for func_name in dir(single_obj)
        if not func_name.startswith("_") and func_name != "np"
    }
    return funcs

def is_optimizer(opt: str) -> str:
    opts = get_optimizer_map().keys()
    if opt not in opts:
        raise argparse.ArgumentTypeError(f"optimizer not found: \"{opt}\"")
    return opt

def is_objective_func(f: str) -> str:
    obj_funcs = get_pyswarms_obj_funcs()
    if f not in obj_funcs:
        raise argparse.ArgumentTypeError(f"objective function not found: \"{f}\"")
    return f
