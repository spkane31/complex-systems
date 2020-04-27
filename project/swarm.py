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
        self.globalBest = [10e9] * dims
        self.historicalBests = [self.globalBest]
        self.evaluations = [] # THIS IS IMPORTANT, SAVES THE GLOBAL BEST FOR EACH RUN

        self.w = 0.9 # inertia
        self.C = 0.5 # Cognitive potential
        self.S = 0.3 # Social Potential

        self.lower_bound = bounds[0] 
        self.upper_bound = bounds[1] 
        self.iterations = 0
        self.epsilon = 0.005

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
        self.evaluations = [0] * epochs
        for i in range(epochs):
            self.SingleIteration()

            self.evaluations[i] = self.fitnessFunc(self.globalBest)
            self.historicalBests.append(self.globalBest)

            self.UpdateVelocities()
            self.UpdatePositions()

            # Policies
            self.SwapParticles()
            self.SwapVelocities()         
            self.DecreaseVelocity() 
            self.AddParticle()
            self.ReplaceParticle()

            if self.CheckConvergence():
                for j in range(i, epochs):
                    self.evaluations[j] = self.evaluations[i]
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
        return

    def SwapVelocities(self):
        if random.random() < self.velocities:
            i1 = int(random.random() * len(self.particles))
            i2 = int(random.random() * len(self.particles))
            while i2 == i1:
                i2 = int(random.random() * len(self.particles))

            temp = self.particles[i1].currentVel
            self.particles[i1].currentVel = self.particles[i2].currentVel
            self.particles[i2].currentVel = temp
        return

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

    def CorrectlyConverged(self, epsilon=0.01) -> bool:
        """
        Checks that the global minimum found by the swarm is relatively close to the
        actual global minimum, can be adjusted by increasing or decreasing epsilon
        """
        # correctMin = fit.actual_minimum(self._fitnessString, self.dimensions)
        # euc_dist = 0
        # for i in range(self.dimensions):
        #     euc_dist += pow(correctMin[i] - self.globalBest[i], 2)

        # euc_dist = pow(euc_dist, 0.5)
        # print(self.fitnessFunc(self.globalBest), self.fitnessFunc(fit.actual_minimum(self._fitnessString, self.dimensions)))
        diff = self.fitnessFunc(self.globalBest) - self.fitnessFunc(fit.actual_minimum(self._fitnessString, self.dimensions))
        # print(diff)
        return abs(diff) < epsilon
        
        # return euc_dist < 0.01

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
    

def increasingDimensions(fxn, iterations, epochs, particles):
    """
    See how an increasing number of dimensions affects convergence rate
    """
    print(f"Function: {fxn}\nIterations: {iterations}\nEpochs: {epochs}\nParticles: {particles}\n")

    dimension_range = 13
    bounds = fit.bounds[fxn]
    
    m_convergence = [0] * 6
    for i in range(6):
        m_convergence[i] = []

    p = 0.20
    for dim in range(2, 2 + dimension_range + 1):
        print(f"Working on {dim} dimensions")
        convergence_rate = [0] * 6
        for iteration in range(iterations):
            s = Swarm(particles, dim, f, bounds)
            i = s.Run(epochs)
            if s.CorrectlyConverged():
                convergence_rate[0] += 1

            s = Swarm(particles, dimensions, f, bounds, swapping=p)
            i = s.Run(epochs)
            if s.CorrectlyConverged():
                convergence_rate[1] += 1

            s = Swarm(particles, dimensions,  f, bounds, velocities=p)
            i = s.Run(epochs)
            if s.CorrectlyConverged():
                convergence_rate[2] += 1
            
            s = Swarm(particles, dimensions,  f, bounds, decrease_velocity=p)
            i = s.Run(epochs)
            if s.CorrectlyConverged():
                convergence_rate[3] += 1
            
            s = Swarm(particles, dimensions, f, bounds, add_particle=p)
            i = s.Run(epochs)
            if s.CorrectlyConverged():
                convergence_rate[4] += 1
            
            s = Swarm(particles, dimensions, f, bounds, replace_particle=p)
            i = s.Run(epochs)
            if s.CorrectlyConverged():
                convergence_rate[5] += 1

        temp = [0] * 6
        for i in range(len(convergence_rate)):
            temp[i] = convergence_rate[i] / iterations

        for i in range(len(temp)):
            m_convergence[i].append(temp[i])
        print(temp)

    fig = plt.figure()
    ax = plt.subplot(111)
    x_axis = np.arange(2, 2+dimension_range+1)
    labels = [
        "Classic PSO",
        "Swapping Positions",
        "Swap Velocities",
        "Decrease Velocities",
        "Add Particle",
        "Replace Particle"
    ]

    for i in range(len(m_convergence)):
        ax.plot(x_axis, m_convergence[i], label=labels[i])

    plt.xlabel('Dimensions')
    plt.ylabel('Percent Converged Correctly')
    plt.title(f'Correct Convergence As Dimensions Increase For {fxn} Function')
    chartBox = ax.get_position()
    ax.set_position([chartBox.x0, chartBox.y0, chartBox.width * 0.6, chartBox.height])
    ax.legend(loc="upper center", bbox_to_anchor=(1.45, 0.8), shadow=True, ncol=1)
    
    plt.show()
           

    return

if __name__ == "__main__":
    args = process_arguments()
    epochs = args.epochs
    iterations = args.iterations
    particles = args.particles
    f = args.fitness_func
    dimensions = args.dimensions

    bounds = fit.bounds[args.fitness_func]

    increasingDimensions(f, iterations, epochs, particles)

    quit()

    num_policies = 6

    stats = []
    for i in range(num_policies):
        stats.append([0] * iterations)
    convergeCount = [0] * num_policies
    time_to_run = [0] * num_policies

    bestValuePerIteration = [[]] * num_policies

    for iteration in range(iterations):
        if iteration % 10 == 0: print(f'Iteration {iteration}')
        if args.run_classic or args.run_all:
            start = time.time()
            s = Swarm(particles, dimensions, f, bounds)
            i = s.Run(epochs)
            e = s.evaluations
            if s.CorrectlyConverged():
                convergeCount[0] += 1
            time_to_run[0] += time.time() - start
            stats[0][iteration] = i
            bestValuePerIteration[0].append(e)

        if args.run_pos_swap or args.run_all:
            start = time.time()
            s = Swarm(particles, dimensions, f, bounds, swapping=0.20)
            i = s.Run(epochs)
            e2 = s.evaluations
            if s.CorrectlyConverged():
                convergeCount[1] += 1
            time_to_run[1] += time.time() - start
            stats[1][iteration] = i
            bestValuePerIteration[1].append(e2)
        
        if args.run_vel_swap or args.run_all:
            start = time.time()
            s = Swarm(particles, dimensions,  f, bounds, velocities=0.20)
            i = s.Run(epochs)
            e3 = s.evaluations
            if s.CorrectlyConverged():
                convergeCount[2] += 1
            time_to_run[2] += time.time() - start
            stats[2][iteration] = i
            bestValuePerIteration[2].append(e3)

        if args.run_dec_vel or args.run_all:
            start = time.time()
            s = Swarm(particles, dimensions,  f, bounds, decrease_velocity=0.20)
            i = s.Run(epochs)
            e4 = s.evaluations
            if s.CorrectlyConverged():
                convergeCount[3]+= 1
            time_to_run[3] += time.time() - start
            stats[3][iteration] = i
            bestValuePerIteration[3].append(e4)

        if args.run_add_particle or args.run_all:
            start = time.time()
            s = Swarm(particles, dimensions, f, bounds, add_particle=0.20)
            i = s.Run(epochs)
            e5 = s.evaluations
            if s.CorrectlyConverged():
                convergeCount[4] += 1
            time_to_run[4] += time.time() - start
            stats[4][iteration] = i
            bestValuePerIteration[4].append(e5)

        if args.run_replace_particle or args.run_all:
            start = time.time()
            s = Swarm(particles, dimensions, f, bounds, replace_particle=0.20)
            i = s.Run(epochs)
            e6 = s.evaluations
            if s.CorrectlyConverged():
                convergeCount[5] += 1
            time_to_run[5] += time.time() - start
            stats[5][iteration] = i
            bestValuePerIteration[5].append(e6)
    

    labels = ["Classic", "Swapping Position", "Swapping Velocities", "Decrease Velocities", "Add Particle", "Replace Particle"]
    print("\nStats")
    for i in range(len(stats)):
        print(labels[i])
        print(f"\tCorrect convergence rate: {100 * convergeCount[i] / iterations} %.")
        print(f"\tConverged in {round(sum(stats[i]) / iterations, 2)} iterations on average. Std. dev: {round(statistics.stdev(stats[i]), 2)}")
        print(f"\tTime to converge {round(time_to_run[i] / iteration, 5)} seconds.\n")
    
    boxAndWhiskerPlot(stats)
    
    # Plotting
    fig = plt.figure()
    ax = plt.subplot(111)

    x_axis = np.arange(1, epochs)

    if args.run_classic:
        ax.plot(e, label="Classic PSO")
    if args.run_pos_swap:
        ax.plot(e2, label="Swapping Positions")
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