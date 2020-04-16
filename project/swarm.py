from scratch import Particle
from params import swarmNames
import fit
import random

import matplotlib.pyplot as plt
import numpy as np

class Swarm():
    def __init__(self, size: int, fitness, swapping=False):
        self.particles = []
        for i in range(size): 
            self.particles.append(Particle.New(swarmNames[i]))
        self.fitnessFunc = fitness
        self.globalBest = [10e9, 10e9]
        self.historicalBests = [self.globalBest]
        self.evaluations = [10]

        self.w = 0.9 # inertia
        self.C = 0.5 # Cognitive potential
        self.S = 0.3 # Social Potential
        self.S2 = 0.3 

        self.upper_bound = [5.8, 5.8]
        self.lower_bound = [-5.8, -5.8]
        self.iterations = 0
        self.epsilon = 0.001

        self.swapping = True

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
        for _ in range(iterations):
            self.SingleIteration()
            self.UpdateVelocities()
            self.UpdatePositions()

            if self.swapping:
                self.SwapParticles()

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
            # p.currentPos += p.currentVel

            # Check Upper Bounds
            if p.currentPos[0] > self.upper_bound[0]:
                p.currentPos[0] = self.upper_bound[0]
            if p.currentPos[1] > self.upper_bound[1]:
                p.currentPos[1] = self.upper_bound[1]
            # Check Lower Bounds
            if p.currentPos[0] < self.lower_bound[0]:
                p.currentPos[0] = self.lower_bound[0]
            if p.currentPos[1] < self.lower_bound[1]:
                p.currentPos[1] = self.lower_bound[1]

        
        pass

    def SwapParticles(self):
        i1 = int(random.random() * len(self.particles))
        i2 = int(random.random() * len(self.particles))
        while i2 != i1:
            i2 = int(random.random() * len(self.particles))

        temp = self.particles[i1].currentPos
        self.particles[i1].currentPos = self.particles[i2].currentPos
        self.particles[i2].currentPos = temp
        return
            
    def CheckConvergence(self):
        """
        Checks the last ten global bests euclidean distances to the previous iterations

        If this value is less than self.epsilon, the swarm has converged.
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

if __name__ == "__main__":
    epochs = 10000
    s = Swarm(20, fit.rastrigin)

    s.Run(epochs)

    e = s.evaluations[1:]
    s = Swarm(20, fit.rastrigin, swapping=True)
    s.Run(epochs)

    fig = plt.figure()
    ax = plt.subplot(111)

    x_axis = np.arange(1, epochs)
    ax.plot(e, label="No Swapping")
    ax.plot(s.evaluations[1:], label="Swapping")

    # plt.plot(e)
    # plt.plot(s.evaluations[1:])
    plt.xlabel("Epochs")
    plt.ylabel("Happiness")
    chartBox = ax.get_position()
    ax.set_position([chartBox.x0, chartBox.y0, chartBox.width * 0.6, chartBox.height])
    ax.legend(loc="upper center", bbox_to_anchor=(1.45, 0.8), shadow=True, ncol=1)

    plt.show()