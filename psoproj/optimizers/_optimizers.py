from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Type

import pyswarms
from pyswarms.base import SwarmOptimizer

from . import mutagenic_pso

def get_optimizer_map() -> Dict[str, Type[SwarmOptimizer]]:
    opts = {
        "local_best": pyswarms.local_best.LocalBestPSO,
        "global_best": pyswarms.global_best.GlobalBestPSO,
        "mutagenic": mutagenic_pso.MutagenicPSO
    }
    return opts
