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


def metafitness(position, size=20, dims=10, iters=5, epochs=100, task_names=None):
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

    task_names = task_names if task_names is not None else ["rastrigin", "ackley", "sphere"]

    task_errors = []
    for itask, task_name in enumerate(task_names):
        task = fit.string_to_func[task_name]
        task_bounds = fit.bounds[task_name]
        task_best_loc = fit.actual_minimum(task_name, dims)
        task_best_val = task(task_best_loc)

        iterations_errors = []
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
            result = s.Run(epochs=epochs)
            swarm_best_key = sorted(list(result["global_bests"].keys()))[-1]
            swarm_best_loc = result["global_bests"][swarm_best_key]
            swarm_best_val = task(swarm_best_loc)

            error = (task_best_val - swarm_best_val) ** 2
            iterations_errors.append(error)
        avg_iterations_error = sum(iterations_errors) / len(iterations_errors)
        task_errors.append(avg_iterations_error)
    
    # now we have a list (per-task) of average errors.
    # combine *those* with an average (TODO: maybe some consistent scaling?)

    # this is our final "meta"-fitness we are trying to optimize.
    avg_task_error = sum(task_errors) / len(task_errors)
    return avg_task_error

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
