import numpy as np
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
from os import system, name
from time import sleep

def clear():
    _ = system('clear')

clear()
print('\nGlobal\n')

# Create bounds
max_bound = 5.12 * np.ones(2)
min_bound = - max_bound
bounds = (min_bound, max_bound)

# Initialize swarm
options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}

# Call instance of PSO with bounds argument
optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=2, options=options, bounds=bounds)

# Perform optimization
cost1, pos1 = optimizer.optimize(fx.rastrigin, iters=10000)

sleep(1)
clear()

print('\nLocal\n')

options = {'c1': 0.5, 'c2': 0.3, 'w':0.9, 'k':2, 'p': 2}

# Call instance of PSO with bounds argument
optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=2, options=options, bounds=bounds)

# Perform optimization
cost2, pos2 = optimizer.optimize(fx.rastrigin, iters=10000)

sleep(1)
clear()

print(f"\nGlobal PSO best position:{pos1}")
print(f"\nLocal PSO best position:{pos2}\n")
