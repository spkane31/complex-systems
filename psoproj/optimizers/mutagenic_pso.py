import numpy as np
import pyswarms
from pyswarms.base import SwarmOptimizer

class MutagenicPSO(SwarmOptimizer):
    def __init__(self):
        super(MutagenicPSO, self).__init__()
