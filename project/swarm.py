from particle import Particle
from params import swarmNames
import fit
import random
import copy
import argparse
import time
import statistics

import matplotlib.pyplot as plt
import numpy as np

class Swarm():
    def __init__(
        self,
        size: int,
        dims: int,
        fitness,
        bounds: tuple,
        swapping=False,
        velocities=False, 
        decrease_velocity=False,
        add_particle=0,
        replace_particle= False
    ):

        self.dimensions = dims
        self.bounds = bounds
        self.particles = []
        for _ in range(size): 
            self.particles.append(Particle(dims, bounds))
        
        self.fitnessFunc = fit.string_to_func[fitness]
        self._fitnessString = fitness
        self.globalBest = [10e9] * dims
        self.historicalBests = [self.globalBest]
        self.evaluations = []

        self.w = 0.5 # inertia
        self.C = 0.4 # Cognitive potential
        self.S = 0.2 # Social Potential
        # self.S2 = 0.3 

        self.lower_bound = bounds[0] 
        self.upper_bound = bounds[1] 
        self.iterations = 0
        self.epsilon = 0.001

        self.swapping = swapping
        self.velocities = velocities
        self.decrease_velocity = decrease_velocity
        self.add_particle = add_particle
        self.replace_particle = replace_particle

        self.PRINTS = False

    def UpdateGlobal(self, newGlobal: float):
        for p in self.particles:
            p.updateGlobal(newGlobal)

    def SingleIteration(self):
        for p in self.particles:
            val = self.fitnessFunc(p.currentPos)
            
            if p.IsLocalBest(val, self.fitnessFunc):
                p.selfBest = p.currentPos

            if self.IsGlobalBest(val):
                # print(f"\tNew Low: {val}\tOld Low {self.fitnessFunc(self.globalBest)}\n")
                self.globalBest = copy.deepcopy(p.currentPos)
                # print(f"\tSingle Iter: {val, self.fitnessFunc(self.globalBest)}\n")
                # print(f"Position: {self.globalBest}")
        self.iterations += 1

    def Run(self, epochs=100):
        i = 0
        for i in range(epochs):
            self.SingleIteration()

            self.evaluations.append(self.fitnessFunc(self.globalBest))
            self.historicalBests.append(self.globalBest)

            self.UpdateVelocities()
            self.UpdatePositions()

            if self.swapping:
                self.SwapParticles()

            if self.velocities:
                self.SwapVelocities()         

            if self.decrease_velocity:
                self.DecreaseVelocity(1.2) 

            if self.add_particle != 0:
                self.particles.append(Particle(self.dimensions, self.bounds))

            if self.replace_particle:
                self.ReplaceParticle()

            if self.CheckConvergence():
                return i

        return epochs

    def IsGlobalBest(self, val) -> bool:
        if val < self.fitnessFunc(self.globalBest):
            return True

        return False
    
    def UpdateVelocities(self):
        for p in self.particles:
            for i in range (len(p.currentPos)):

                r1 = random.random()
                r2 = random.random()

                vel_cog = self.C * r1 * (p.selfBest[i] - p.currentPos[i])
                vel_soc = self.S * r2 * (self.globalBest[i] - p.currentPos[i])

                p.currentVel[i] = self.w * p.currentVel[i] + vel_cog + vel_soc
        return

    def UpdatePositions(self):
        for p in self.particles:
            for i in range(len(p.currentPos)):
                p.currentPos[i] += p.currentVel[i]
            p.CheckBounds(self.upper_bound, self.lower_bound)    
        return

    def SwapParticles(self, p=0.10):
        # Happens with a probability p
        if random.random() < p:
            i1 = int(random.random() * len(self.particles))
            i2 = int(random.random() * len(self.particles))
            while i2 == i1:
                i2 = int(random.random() * len(self.particles))

            temp = self.particles[i1].currentPos
            self.particles[i1].currentPos = self.particles[i2].currentPos
            self.particles[i2].currentPos = temp
        return

    def SwapVelocities(self, p=0.10):
        if random.random() < p:
            i1 = int(random.random() * len(self.particles))
            i2 = int(random.random() * len(self.particles))
            while i2 == i1:
                i2 = int(random.random() * len(self.particles))

            temp = self.particles[i1].currentVel
            self.particles[i1].currentVel = self.particles[i2].currentVel
            self.particles[i2].currentVel = temp
        return

    def DecreaseVelocity(self, scale, p=0.25):
        if random.random() < p:
            i1 = int(random.random() * len(self.particles))

            for i in range(len(self.particles[i1].currentVel)):
                self.particles[i1].currentVel[i] *= (1.0 / scale)
            
    def ReplaceParticle(self, p=0.25):
        if random.random() < p:
            i = int(random.random() * len(self.particles))
            self.particles[i] = Particle(self.dimensions, self.bounds)
    
    def CheckConvergence(self):
        """
        Checks the last ten global bests euclidean distances to the previous iterations

        If this value is less than self.epsilon, the swarm has converged.

        Not the best way to do it, but a start
        """

        total = 0
        for i in range(1, min(len(self.historicalBests), 10)):
            temp = 0
            for j in range(len(self.historicalBests[0])):

                temp += (self.historicalBests[-i][j] - self.historicalBests[-i-1][j]) ** 2
            total += temp ** 0.5

        if self.PRINTS: 
            print(total)
        if total < self.epsilon:
            return True
        return False

    def CorrectlyConverged(self, epsilon=0.001) -> bool:
        """
        Checks that the global minimum found by the swarm is relatively close to the
        actual global minimum, can be adjusted by increasing or decreasing epsilon
        """
        correctMin = fit.actual_minimum(self._fitnessString, self.dimensions)
        euc_dist = 0
        for i in range(self.dimensions):
            euc_dist += pow(correctMin[i] - self.globalBest[i], 2)

        euc_dist = pow(euc_dist, 0.5)
        
        return euc_dist < 0.001


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

if __name__ == "__main__":
    args = process_arguments()
    epochs = args.epochs
    iterations = args.iterations
    particles = args.particles
    f = fit.string_to_func[args.fitness_func]
    f = args.fitness_func
    dimensions = args.dimensions

    bounds = fit.bounds[args.fitness_func]

    num_policies = 6

    stats = []
    for i in range(num_policies):
        stats.append([0] * iterations)
    convergeCount = [0] * num_policies
    time_to_run = [0] * num_policies
    for iteration in range(iterations):
        print(f"Iteration #{iteration}")

        if args.run_classic or args.run_all:
            start = time.time()
            s = Swarm(particles, dimensions, f, bounds)
            i = s.Run(epochs)
            e = s.evaluations
            if s.PRINTS: print(f(s.globalBest))
            if s.CorrectlyConverged():
                convergeCount[0] += 1
            time_to_run[0] += time.time() - start
            stats[0][iteration] = i

        if args.run_pos_swap or args.run_all:
            start = time.time()
            s = Swarm(particles, dimensions, f, bounds, swapping=True)
            i = s.Run(epochs)
            e2 = s.evaluations
            if s.PRINTS: print(f(s.globalBest))
            if s.CorrectlyConverged():
                convergeCount[1] += 1
            time_to_run[1] += time.time() - start
            stats[1][iteration] = i
        
        if args.run_vel_swap or args.run_all:
            start = time.time()
            s = Swarm(particles, dimensions,  f, bounds, velocities=True)
            i = s.Run(epochs)
            e3 = s.evaluations
            if s.PRINTS: print(f(s.globalBest))
            if s.CorrectlyConverged():
                convergeCount[2] += 1
            time_to_run[2] += time.time() - start
            stats[2][iteration] = i

        if args.run_dec_vel or args.run_all:
            start = time.time()
            s = Swarm(particles, dimensions,  f, bounds, decrease_velocity=True)
            i = s.Run(epochs)
            e4 = s.evaluations
            if s.PRINTS: print(f(s.globalBest))
            if s.CorrectlyConverged():
                convergeCount[3]+= 1
            time_to_run[3] += time.time() - start
            stats[3][iteration] = i

        if args.run_add_particle or args.run_all:
            start = time.time()
            s = Swarm(particles, dimensions, f, bounds, add_particle=20)
            i = s.Run(epochs)
            e5 = s.evaluations
            if s.PRINTS: print(f(s.globalBest))
            if s.CorrectlyConverged():
                convergeCount[4] += 1
            time_to_run[4] += time.time() - start
            stats[4][iteration] = i

        if args.run_replace_particle or args.run_all:
            start = time.time()
            s = Swarm(particles, dimensions, f, bounds, replace_particle=True)
            i = s.Run(epochs)
            e6 = s.evaluations
            if s.PRINTS: print(f(s.globalBest))
            if s.CorrectlyConverged():
                convergeCount[5] += 1
            time_to_run[5] += time.time() - start
            stats[5][iteration] = i
    

    print("\nStats")
    for i in range(len(stats)):
        print(f"\t{100 * convergeCount[i] / iterations} %.")
        print(f"\tConverged in {sum(stats[i]) / iterations} iterations on average. Std. dev: {round(statistics.stdev(stats[i]), 2)}")
        print(f"\tTime to converge {round(time_to_run[i] / iteration, 2)} seconds.")
    

    # Plotting
    fig = plt.figure()
    ax = plt.subplot(111)

    x_axis = np.arange(1, epochs)

    if args.run_classic:
        ax.plot(e, label="Classic PSO")
    if args.run_pos_swap:
        ax.plot(e2, label="Swapping")
    if args.run_vel_swap:
        ax.plot(e3, label="Swap Velocities")
    if args.run_dec_vel:
        ax.plot(e4, label="Decrease Velocities")
    if args.run_add_particle:
        ax.plot(e5, label="Add Particle")
    if args.run_replace_particle or args.run_all:
        ax.plot(e6, label="Replace Particle")


    plt.xlabel("Epochs")
    plt.ylabel("Value")
    chartBox = ax.get_position()
    ax.set_position([chartBox.x0, chartBox.y0, chartBox.width * 0.6, chartBox.height])
    ax.legend(loc="upper center", bbox_to_anchor=(1.45, 0.8), shadow=True, ncol=1)

    plt.show()