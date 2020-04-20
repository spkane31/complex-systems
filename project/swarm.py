from particle import Particle
from params import swarmNames
import fit
import random
import argparse

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
        increase_velocity=False, 
        add_particle=0
    ):

        self.dimensions = dims
        self.bounds = bounds
        self.particles = []
        for i in range(size): 
            self.particles.append(Particle(dims, bounds))
        self.fitnessFunc = fit.string_to_func[fitness]
        self._fitnessString = fitness
        self.globalBest = [10e9, 10e9]
        self.historicalBests = [self.globalBest]
        self.evaluations = [10]

        self.w = 0.9 # inertia
        self.C = 0.5 # Cognitive potential
        self.S = 0.3 # Social Potential
        # self.S2 = 0.3 

        self.lower_bound = bounds[0] 
        self.upper_bound = bounds[1] 
        self.iterations = 0
        self.epsilon = 0.001

        self.swapping = swapping
        self.velocities = velocities
        self.increase_velocity = increase_velocity
        self.add_particle = add_particle


        self.PRINTS = False

    def UpdateGlobal(self, newGlobal: float):
        for p in self.particles:
            p.updateGlobal(newGlobal)

    def SingleIteration(self):
        if self.PRINTS: 
            print("Iteration #", self.iterations)

        for p in self.particles:
            # print("Positions: ", p.currentPos)
            # print("Velocities: ", p.currentVel)
            val = self.fitnessFunc(p.currentPos)
            
            if p.IsLocalBest(val, self.fitnessFunc):
                p.selfBest = p.currentPos

            if self.IsGlobalBest(val):
                self.globalBest = p.currentPos

        
        self.iterations += 1

    def Run(self, iterations=100):
        for i in range(iterations):
            self.SingleIteration()
            self.UpdateVelocities()
            self.UpdatePositions()

            if self.swapping:
                self.SwapParticles()

            if self.velocities:
                self.SwapVelocities()         

            if self.increase_velocity:
                self.IncreaseVelocity(2.0) 

            if self.add_particle != 0:
                if i % self.add_particle == 0:
                    self.particles.append(Particle(self.dimensions, self.bounds))

            self.historicalBests.append(self.globalBest)
            if self.PRINTS: 
                print(f"Best Position: {tuple(self.globalBest)}")

            if self.CheckConvergence():
                return

            self.evaluations.append(self.fitnessFunc(self.globalBest))

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
            while i2 != i1:
                i2 = int(random.random() * len(self.particles))

            temp = self.particles[i1].currentPos
            self.particles[i1].currentPos = self.particles[i2].currentPos
            self.particles[i2].currentPos = temp
        return

    def SwapVelocities(self, p=0.10):
        if random.random() < p:
            i1 = int(random.random() * len(self.particles))
            i2 = int(random.random() * len(self.particles))
            while i2 != i1:
                i2 = int(random.random() * len(self.particles))

            temp = self.particles[i1].currentVel
            self.particles[i1].currentVel = self.particles[i2].currentVel
            self.particles[i2].currentVel = temp
        return

    def IncreaseVelocity(self, scale, p=0.10):
        if random.random() < p:
            i1 = int(random.random() * len(self.particles))

            for i in range(len(self.particles[i1].currentVel)):
                self.particles[i1].currentVel[i] *= scale
            
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
        print(euc_dist)
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
    ap.add_argument("-i", "--iterations", action="store", type=at_least(3), default=10, help="the number of iterations to run. default: %(default)s")
    ap.add_argument("--print-statements", action="store_true", help="enable printing various logging/debug statements.")
    ap.add_argument("--save-images", action="store_true", help="enable saving images of the final space.")
    ap.add_argument("--show", action="store_true", help="show the final plot after program has run.")

    ap.add_argument("--function", required=False, dest="fitness_func", default="rastrigin", help="Fitness function to test on, options are rastrigin, ackley, sphere, styblinski-tang, con, banana. default: %(default)s")
    ap.add_argument("--all", action="store_true", required=False, dest="run_all", help="run all variants. overrides other choices. default: True if no other run options are given, else False")
    ap.add_argument("--swap_vel", action="store_true", required=False, dest="run_vel_swap", help="run the velocity swapping policy. default: %(default)s")
    ap.add_argument("--swap_vos", action="store_true", required=False, dest="run_pos_swap", help="run the position swapping policy. default: %(default)s")
    ap.add_argument("--add_particle", action="store_true", required=False, dest="run_add_particle", help="run the add particle policy. default: %(default)s")
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
    particles = 50
    f = fit.string_to_func[args.fitness_func]
    f = args.fitness_func
    dimensions = 2

    bounds = fit.bounds[args.fitness_func]

    stats = [0] * 5



    # print(f"Minimum: {f(fit.actual_minimum('rastrigin', dimensions))}\n")
    for _ in range(iterations):

        s = Swarm(particles, dimensions, f, bounds)
        s.Run(epochs)
        e = s.evaluations[1:]
        if s.PRINTS: print(f(s.globalBest))
        if s.CorrectlyConverged():
            stats[0] += 1

        s = Swarm(particles, dimensions, f, bounds, swapping=True)
        s.Run(epochs)
        e2 = s.evaluations[1:]
        if s.PRINTS: print(f(s.globalBest))
        if s.CorrectlyConverged():
            stats[1] += 1
        

        s = Swarm(particles, dimensions,  f, bounds, velocities=True)
        s.Run(epochs)
        e3 = s.evaluations[1:]
        if s.PRINTS: print(f(s.globalBest))
        if s.CorrectlyConverged():
            stats[2] += 1

        s = Swarm(particles, dimensions,  f, bounds, increase_velocity=True)
        s.Run(epochs)
        e4 = s.evaluations[1:]
        if s.PRINTS: print(f(s.globalBest))
        if s.CorrectlyConverged():
            stats[3]+= 1

        s = Swarm(particles, dimensions, f, bounds, add_particle=100)
        s.Run(epochs)
        e5 = s.evaluations[1:]
        if s.PRINTS: print(f(s.globalBest))
        if s.CorrectlyConverged():
            stats[4] += 1
        

        # break # TODO: build the tools for averaging over multiple runs

    for stat in stats:
        print(stat / iterations)
    quit()

    # Plotting
    fig = plt.figure()
    ax = plt.subplot(111)

    x_axis = np.arange(1, epochs)
    ax.plot(e, label="Classic PSO")
    ax.plot(e2, label="Swapping")
    ax.plot(e3, label="Swap Velocities")
    ax.plot(e4, label="Increase Velocities")


    # plt.plot(e)
    # plt.plot(s.evaluations[1:])
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    chartBox = ax.get_position()
    ax.set_position([chartBox.x0, chartBox.y0, chartBox.width * 0.6, chartBox.height])
    ax.legend(loc="upper center", bbox_to_anchor=(1.45, 0.8), shadow=True, ncol=1)

    plt.show()