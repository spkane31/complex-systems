import argparse
import json

import fit
from swarm import Swarm


meta_params = [
    "w",
    "C",
    "S",
    "swapping",
    "velocities",
    "decrease_velocity",
    #"add_particle",
    "replace_particle",
]


def metafitness(position, size=20, dims=10, iters=5, epochs=100):
    # unpack position vector
    (
        w,
        C,
        S,
        swapping,
        velocities,
        decrease_velocity,
        #add_particle,
        replace_particle,
    ) = position

    for task_name in ["rastrigin", "ackley", "sphere"]:
        task = fit.string_to_func[task_name]
        task_bounds = fit.bounds[task_name]
        task_best_loc = fit.actual_minimum(task_name, dims)
        task_best_val = fit.string_to_func[task_name](task_best_loc)
        for iter_ in range(iters):
            s = Swarm(
                size,
                dims,
                fitness=task_name,
                bounds=task_bounds,
                w=w,
                C=C,
                S=S,
                swapping=swapping,
                velocities=velocities,
                decrease_velocity=decrease_velocity,
                add_particle=0.0,
                replace_particle=replace_particle
            )
            res = s.Run(epochs=epochs)
        # TODO: finish this by aggregating stats and boiling it all down
        # to a single "fitness" number that is best when minimized.

def _dummy_false(*args, **kwargs):
    return False

def main():
    metaswarm = Swarm(
        size=25, dims=len(meta_params), fitness="sphere", bounds=(0.0, 1.0)
    )
    metaswarm.fitnessFunc = metafitness
    metaswarm._fitnessString = "metafitness"
    metaswarm.CheckConvergence = _dummy_false
    results = metaswarm.Run(100)
    results.save_json("metaswarm_output.json")

    best_settings_key = sorted(list(results["global_bests"].keys()))[-1]
    best_settings = results["global_bests"][best_settings_key]
    for setting, value in zip(meta_params, best_settings):
        print(f"{setting}\t{value}")


if __name__ == "__main__":
    main()
