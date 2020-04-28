import argparse
import copy
import json
import random
import statistics
import time

import matplotlib.pyplot as plt
import numpy as np

import fit
from particle import Particle
from results import RunResults

from typing import Any, Dict, List, Optional

class Swarm():
    def __init__(
        self,
        size: int,
        dims: int,
        fitness,
        bounds: tuple,
        swapping=0.0,
        velocities=0.0, 
        decrease_velocity=0.0,
        add_particle=0.0,
        replace_particle= 0.0
    ):
        self.dimensions = dims
        self.bounds = bounds
        self.particles = []
        for _ in range(size): 
            self.particles.append(Particle(dims, bounds))
        
        self.fitnessFunc = fit.string_to_func[fitness]
        self._fitnessString = fitness

        self.w = 0.5 # inertia
        self.C = 0.4 # Cognitive potential
        self.S = 0.2 # Social Potential

        self.lower_bound = bounds[0] 
        self.upper_bound = bounds[1] 
        self.iterations = 0
        self.epsilon = 0.005

        self.swapping = swapping
        self.velocities = velocities
        self.decrease_velocity = decrease_velocity
        self.add_particle = add_particle
        self.replace_particle = replace_particle

    @classmethod
    def update_particle_velocities(cls, particles, global_best_loc, w, C, S):
        for p in particles:
            for i in range (len(p.currentPos)):
                r1 = random.random()
                r2 = random.random()
                vel_cog = C * r1 * (p.selfBest[i] - p.currentPos[i])
                vel_soc = S * r2 * (global_best_loc[i] - p.currentPos[i])
                p.currentVel[i] = w * p.currentVel[i] + vel_cog + vel_soc
    
    @classmethod
    def update_particle_positions(cls, particles, upper_bound, lower_bound):
        for p in particles:
            for i in range(len(p.currentPos)):
                p.currentPos[i] += p.currentVel[i]
            p.CheckBounds(upper_bound, lower_bound)

    @classmethod
    def update_particles(
        cls, particles, global_best_loc, w, C, S, upper_bound, lower_bound
    ):
        cls.update_particle_velocities(particles, global_best_loc, w, C, S)
        cls.update_particle_positions(particles, upper_bound, lower_bound)
    
    @classmethod
    def evaluate_particles(cls, particles, fitness_func, global_best_loc):
        best_loc = copy.deepcopy(global_best_loc)
        best_fitness = fitness_func(best_loc) if best_loc is not None else float("inf")
        for p in particles:
            val = fitness_func(p.currentPos)
            # update self best
            if val < fitness_func(p.selfBest):
                p.selfBest = copy.deepcopy(p.currentPos)
            # update global best
            if val < best_fitness:
                best_loc = copy.deepcopy(p.currentPos)
                best_fitness = val
        return best_loc, best_fitness

    def Run(self, epochs=100):
        results = RunResults()
        results.add_pso_params(
            self._fitnessString,
            self.dimensions,
            self.w,
            self.C,
            self.S,
            p_swap_pos = self.swapping,
            p_swap_vel = self.velocities,
            p_decrease_vel = self.decrease_velocity,
            p_add_particle = self.add_particle,
            p_replace_particle = self.replace_particle
        )
        start_time = time.time()
        # epoch 0: initial performace / find global best
        global_best_loc, global_best_fitness = self.evaluate_particles(
            self.particles,
            self.fitnessFunc,
            None
        )
        results.add_global_best(0, global_best_loc)
        for e in range(1, epochs+1):
            # Apply policies
            self.SwapParticles()
            self.SwapVelocities()
            self.DecreaseVelocity() 
            self.AddParticle()
            self.ReplaceParticle()

            self.update_particles(
                self.particles,
                global_best_loc,
                self.w,
                self.C,
                self.S,
                self.upper_bound,
                self.lower_bound
            )

            # Evaluate particle positions, update particle and global bests
            global_best_loc, global_best_fitness = self.evaluate_particles(
                self.particles,
                self.fitnessFunc,
                global_best_loc
            )
            results.add_global_best(e, global_best_loc)
            results.epochs += 1

            if self.CheckConvergence():
                break # TODO: maybe fill out rest of results with last data?
        end_time = time.time()
        results.add_runtime(end_time - start_time)
        return results

    def SwapParticles(self):
        # Happens with a probability p
        if random.random() < self.swapping:
            i1 = int(random.random() * len(self.particles))
            i2 = int(random.random() * len(self.particles))
            while i2 == i1:
                i2 = int(random.random() * len(self.particles))

            temp = self.particles[i1].currentPos
            self.particles[i1].currentPos = self.particles[i2].currentPos
            self.particles[i2].currentPos = temp

    def SwapVelocities(self):
        if random.random() < self.velocities:
            i1 = int(random.random() * len(self.particles))
            i2 = int(random.random() * len(self.particles))
            while i2 == i1:
                i2 = int(random.random() * len(self.particles))

            temp = self.particles[i1].currentVel
            self.particles[i1].currentVel = self.particles[i2].currentVel
            self.particles[i2].currentVel = temp

    def DecreaseVelocity(self):
        if random.random() < self.decrease_velocity:
            i1 = int(random.random() * len(self.particles))
            scale = 1 + random.random()
            for i in range(len(self.particles[i1].currentVel)):
                self.particles[i1].currentVel[i] *= (1.0 / scale)
            
    def ReplaceParticle(self):
        if random.random() < self.replace_particle:
            i = int(random.random() * len(self.particles))
            self.particles[i] = Particle(self.dimensions, self.bounds)

    def AddParticle(self):
        if random.random() < self.add_particle:
            self.particles.append(Particle(self.dimensions, self.bounds))
    
    def CheckConvergence(self):
        """
        If all the particles are very close to each other, the particles have converged on each other

        This is exponential time to calculate all the distances.
        """
        for i in range(len(self.particles)):
            for j in range(i+1, len(self.particles)):
                d = self.EuclideanDistance(self.particles[i].currentPos, self.particles[j].currentPos)
                if d > self.epsilon:
                    return False
        return True

    @staticmethod
    def CorrectlyConverged(results, epsilon=0.001) -> bool:
        """
        Checks that the global minimum found by the swarm is relatively close to the
        actual global minimum, can be adjusted by increasing or decreasing epsilon
        """
        dim = results["pso_params"]["dimension"]
        fitness_func_name = results["pso_params"]["fitness_function"]
        last_epoch = sorted(list(results["global_bests"].keys()))[-1]
        global_best_location = results["global_bests"][last_epoch]

        correctMin = fit.actual_minimum(fitness_func_name, dim)
        euc_dist = 0.0
        for i in range(dim):
            euc_dist += pow(correctMin[i] - global_best_location[i], 2)
        euc_dist = pow(euc_dist, 0.5)
        return euc_dist < epsilon

    def EuclideanDistance(self, loc1, loc2):
        sums = 0
        for i in range(len(loc1)):
            sums += (loc1[i] - loc2[i]) ** 2
        return sums ** 0.5


def at_least(k: int):
    def at_least_fn(n: str):
        try:
            n_int = int(n)
            if not n_int >= k:
                raise ValueError()
            return n_int
        except:
            raise argparse.ArgumentTypeError(f"invalid value {n}; expected positive integer >= {k}")
    return at_least_fn

def get_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    # fmt: off
    ap.add_argument("-e", "--epochs", action="store", type=at_least(1), default=30, help="the number of epochs to run. default: %(default)s")
    ap.add_argument("-i", "--iterations", action="store", type=at_least(1), default=10, help="the number of iterations to run. default: %(default)s")
    ap.add_argument("-d", "--dimensions", action="store", type=at_least(2), default=2, help="Dimensions to test PSO in. default: %(default)s")
    ap.add_argument("-p", "--particles", action="store", type=at_least(3), default=10, help="Number of particles to use in a simulation. default: %(default)s")
    ap.add_argument("--print-statements", action="store_true", help="enable printing various logging/debug statements.")
    ap.add_argument("--save-images", action="store_true", help="enable saving images of the final space.")
    ap.add_argument("--show", action="store_true", help="show the final plot after program has run.")

    ap.add_argument("--function", required=False, dest="fitness_func", default="rastrigin", help="Fitness function to test on, options are rastrigin, ackley, sphere, styblinski-tang, con, banana. default: %(default)s")
    ap.add_argument("--all", action="store_true", required=False, dest="run_all", help="run all variants. overrides other choices. default: True if no other run options are given, else False")
    ap.add_argument("--classic", action="store_true", required=False, dest="run_classic", help="run the classic variant of PSO. default: %(default)s")
    ap.add_argument("--swap_vel", action="store_true", required=False, dest="run_vel_swap", help="run the velocity swapping policy. default: %(default)s")
    ap.add_argument("--swap_pos", action="store_true", required=False, dest="run_pos_swap", help="run the position swapping policy. default: %(default)s")
    ap.add_argument("--dec_vel", action="store_true", required=False, dest="run_dec_vel", help="run the velocity decreasing policy. default: %(default)s")
    ap.add_argument("--add_particle", action="store_true", required=False, dest="run_add_particle", help="run the add particle policy. default: %(default)s")
    ap.add_argument("--replace_particle", action="store_true", required=False, dest="run_replace_particle", help="run the replace particle policy. default: %(default)s")

    ap.add_argument("output", metavar="OUTPUT_FILE", action="store")
    # fmt: on
    return ap

def process_arguments() -> argparse.Namespace:
    args = get_argparser().parse_args()
    none_selected = not any(
        [
            arg_setting
            for arg_name, arg_setting in vars(args).items()
            if arg_name.startswith("run_")
        ]
    )
    args.run_all |= none_selected

    # set all run settings to true if run_all was specified
    if args.run_all == True:
        args_dict = vars(args)
        for run_arg in [k for k in args_dict.keys() if k.startswith("run_")]:
            args_dict[run_arg] = True

    return args

def boxAndWhiskerPlot(data: list):
    # Plot a box and whisker of each strategy
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_title('Comparing Different PSO Strategies')
    ax1.set_xlabel('PSO Strategies')
    ax1.set_ylabel('Iterations')
    bp = ax1.boxplot(data)
    fig.show()
    

def increasingDimensions():
    """
    See how an increasing number of dimensions affects iterations to 
    converge for each type.
    """

if __name__ == "__main__":
    args = process_arguments()
    epochs = args.epochs
    iterations = args.iterations
    particles = args.particles
    f = args.fitness_func
    dimensions = args.dimensions

    bounds = fit.bounds[args.fitness_func]

    num_policies = 6

    stats = []
    for i in range(num_policies):
        stats.append([0] * iterations)
    convergeCount = [0] * num_policies
    time_to_run = [0] * num_policies

    bestValuePerIteration = [[]] * num_policies

    swarm_results: Dict[str, Any] = {}
    for iteration in range(iterations):
        print(f'Iteration {iteration+1}')

        if args.run_classic or args.run_all:
            KEY = "classic"
            swarm_results.setdefault(KEY, [])
            results = (
                Swarm(particles, dimensions, f, bounds)
                .Run(epochs)
            )
            swarm_results[KEY].append(results.as_dict())

        if args.run_pos_swap or args.run_all:
            KEY = "pos_swap"
            swarm_results.setdefault(KEY, [])
            results = (
                Swarm(particles, dimensions, f, bounds, swapping=0.20)
                .Run(epochs)
            )
            swarm_results[KEY].append(results.as_dict())
        
        if args.run_vel_swap or args.run_all:
            KEY = "vel_swap"
            swarm_results.setdefault(KEY, [])
            results = (
                Swarm(particles, dimensions,  f, bounds, velocities=0.20)
                .Run(epochs)
            )
            swarm_results[KEY].append(results.as_dict())

        if args.run_dec_vel or args.run_all:
            KEY = "dec_vel"
            swarm_results.setdefault(KEY, [])
            results = (
                Swarm(particles, dimensions,  f, bounds, decrease_velocity=0.20)
                .Run(epochs)
            )
            swarm_results[KEY].append(results.as_dict())

        if args.run_add_particle or args.run_all:
            KEY = "add_particle"
            swarm_results.setdefault(KEY, [])
            results = (
                Swarm(particles, dimensions, f, bounds, add_particle=0.20)
                .Run(epochs)
            )
            swarm_results[KEY].append(results.as_dict())

        if args.run_replace_particle or args.run_all:
            KEY = "replace_particle"
            swarm_results.setdefault(KEY, [])
            results = (
                Swarm(particles, dimensions, f, bounds, replace_particle=0.20)
                .Run(epochs)
            )
            swarm_results[KEY].append(results.as_dict())
    
    with open(args.output, "w") as resultsf:
        json.dump(swarm_results, resultsf)

    # labels = ["Classic", "Swapping Position", "Swapping Velocities", "Decrease Velocities", "Add Particle", "Replace Particle"]
    # print("\nStats")
    # for i in range(len(stats)):
    #     print(labels[i])
    #     print(f"\tCorrect convergence rate: {100 * convergeCount[i] / iterations} %.")
    #     print(f"\tConverged in {round(sum(stats[i]) / iterations, 2)} iterations on average. Std. dev: {round(statistics.stdev(stats[i]), 2)}")
    #     print(f"\tTime to converge {round(time_to_run[i] / iteration, 5)} seconds.\n")
    
    # boxAndWhiskerPlot(stats)
    
    # # Plotting
    # fig = plt.figure()
    # ax = plt.subplot(111)

    # x_axis = np.arange(1, epochs)

    # if args.run_classic:
    #     ax.plot(e, label="Classic PSO")
    # if args.run_pos_swap:
    #     ax.plot(e2, label="Swapping Positions")
    # if args.run_vel_swap:
    #     ax.plot(e3, label="Swap Velocities")
    # if args.run_dec_vel:
    #     ax.plot(e4, label="Decrease Velocities")
    # if args.run_add_particle:
    #     ax.plot(e5, label="Add Particle")
    # if args.run_replace_particle or args.run_all:
    #     ax.plot(e6, label="Replace Particle")


    # plt.xlabel("Epochs")
    # plt.ylabel("Value")
    # chartBox = ax.get_position()
    # ax.set_position([chartBox.x0, chartBox.y0, chartBox.width * 0.6, chartBox.height])
    # ax.legend(loc="upper center", bbox_to_anchor=(1.45, 0.8), shadow=True, ncol=1)

    # plt.show()
