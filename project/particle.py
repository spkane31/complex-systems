# Bayley King
from os import system, name
from time import sleep
import timeit
import random
import numpy as np
import fit
from params import *
np.set_printoptions(precision=3)

from typing import List

def clear():
    _ = system('clear')

#clear()

class Particle():
    def __init__(self, dimensions, bounds):
        if bounds[0] == 0:
            bounds = (10, 10)
        # self.name = name
        # self.globalBest = randomInit()
        self.localBest = randomInit()
        self.currentPos = [0] * dimensions
        for d in range(dimensions):
            self.currentPos[d] = random.uniform(-1, 1) * bounds[0]
        self.localBest = self.currentPos
        self.selfBest = self.currentPos
        self.currentVel = [random.random()] * dimensions
        self.formattedList = []

    def printVals(self):
        self.formattedList = []
        for item in self.output:
            for i in item:
                #print(i)
                self.formattedList.append("%.2f"%i)        
        # print(self.name)
        # print('\tGlobal Best:',self.globalBest)
        print('\tLocal Best:',self.localBest)
        print('\tSelf Best:',self.selfBest)
        print('\tCurrent Location:',self.currentPos)
        print('\tCurrent Velocity:',self.currentVel)
        print("Fitness:{0:.3f}".format(fit.rastrigin(self.currentPos)))

    def printPos(self):
        self.formattedList = []
        for item in self.output:
            for i in item:
                #print(i)
                self.formattedList.append("%.2f"%i)        
        print('\tLocation:',self.currentPos,"\tFitness:{0:.3f}".format(fit.rastrigin(self.currentPos)))
        
    def IsLocalBest(self, loc, f):
        if loc <= f(self.currentPos):
            return True
        return False

    def CheckBounds(self, upper, lower):
        for i in range(len(self.currentPos)):
            if self.currentPos[i] > upper:
                self.currentPos[i] = upper
            elif self.currentPos[i] < lower:
                self.currentPos[i] = lower

# define the Swarm type
Swarm = List[Particle]
 
def randVal():
    return np.random.random()

def update_velocity(particle):
    v = np.array(particle.currentVel)
    p = np.array(particle.currentPos)

    particle.currentVel = (np.multiply(w,v)) + \
        (np.multiply(C*randVal(),(np.subtract(particle.selfBest,p)))) + \
        (np.multiply(S*randVal(),(np.subtract(particle.globalBest,p)))) #+ \
        #(np.multiply(S2*randVal(),(np.subtract(particle.localBest,p))))

def update_position(particle):
    particle.currentPos += particle.currentVel
    if particle.currentPos[0] >= ub[0]:
        particle.currentPos[0] = ub[0]
    elif particle.currentPos[0] <= lb[0]:
        particle.currentPos[0] = lb[0]
    if particle.currentPos[1] >= ub[1]:
        particle.currentPos[1] = ub[1]
    elif particle.currentPos[1] <= lb[1]:
        particle.currentPos[1] = lb[1]

def updateGlobal(newGlobal: float, swarm: Swarm):
    for particle in swarm:
        particle.globalBest = newGlobal

def randomInit():
    x = np.random.random()*5.8
    y = np.random.random()*5.8
    return x,y

def printParts(swarm):
    [particle.printVals() for particle in swarm]

def distance(pos1,pos2):
    return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

def updateLocal(swarm: Swarm):
    finalDist = []
    for particle1 in swarm:
        dist = []
        for particle2 in swarm:
            if particle1.name == particle2.name:
                continue
            dist.append([distance(particle1.currentPos,particle2.currentPos),particle2.name,fit.rastrigin(particle2.currentPos),particle2.currentPos])
        dist.sort()
        finalDist.append(dist[:n])
    counter = 0
    for d in finalDist:
        d.sort(key = lambda x: x[2])
        swarm[counter].localBest = d[0][3]
        counter += 1
    #print('\n',finalDist,'\n')

def swapPos(swarm):
    fits = []
    for particle in swarm:
        fits.append([fit.rastrigin(particle.currentPos),particle.name,particle.currentPos])
    fits.sort()
    fits = fits[-2:]
    #print('\n',fits,'\n')
    for particle in swarm:
        if particle.name == fits[0][1]:
            particle.currentPos = fits[1][2]
        elif particle.name == fits[1][1]:
            particle.currentPos = fits[0][2]    

def resetPos(swarm):
    fits = []
    for particle in swarm:
        fits.append([fit.rastrigin(particle.currentPos),particle.name,particle.currentPos])
    fits.sort()
    fits = fits[-2:]
    #print('\n',fits,'\n')
    for particle in swarm:
        if particle.name == fits[0][1]:
            particle.currentPos = [np.random.random()*5.8,np.random.random()*5.8]
        elif particle.name == fits[1][1]:
            particle.currentPos = [np.random.random()*5.8,np.random.random()*5.8]

def main():
    swarm = []
    # particle initialization
    for name in swarmNames:
        swarm.append(Particle.New(name))
    
    # finds the best position from the init particle locations
    bestG = 9999
    for particle in swarm:
        if fit.rastrigin(particle.currentPos) < bestG:
           loc = particle.currentPos
           bestG = fit.rastrigin(particle.currentPos)

    for particle in swarm:
        particle.updateGlobal(loc)
    updateGlobal(loc, swarm)

    #printParts(swarm) # [particle.printVals() for particle in swarm] # prints particles
    updateLocal(swarm)
    printParts(swarm)
    swapNums = 0

    start = timeit.default_timer()
    iterations = 10000
    for i in range(iterations):
        for particle in swarm:
            #if i == 0 or i == 100 or i == 1000 or i == 5000:
            #    particle.printPos()
            # Part 1: If current position is less than the personal best,
            if fit.rastrigin(particle.currentPos) < fit.rastrigin(particle.selfBest):
                # Update personal best
                particle.selfBest = particle.currentPos
            # Part 2: If current pos is less than global best,
            if fit.rastrigin(particle.currentPos) < fit.rastrigin(particle.globalBest):
                updateGlobal(swarm,particle.currentPos)
                # Update global best
            # Part 3: If personal best is less than local best,
#            if fit.rastrigin(particle.currentPos) < fit.rastrigin(particle.localBest):
                # Update global best
#                updateLocal(swarm)
            # Part 4: Update velocity and position matrices
            update_velocity(particle)
            update_position(particle)
            #if i % 50== 0:
            #    print('Iterations:',i)            
            #    particle.printVals()
        '''
        if randVal() < p:
            resetPos(swarm)
            swapNums += 1
        '''

    stop = timeit.default_timer()
    print('##################################')
    bestVal = 9999
    for particle in swarm:
        #particle.printVals()
        particle.printPos()
        if fit.rastrigin(particle.currentPos) <= bestVal:
            bestVal = fit.rastrigin(particle.currentPos)
            best = particle
    print('W:',w,'  C:',C,'  S1:',S,'  S2:',S2,'  N:',n,'  p:',p)
    print('Number of swaps:',swapNums)
    #best.printPos()
    print('Time to run: {0:.3f}'.format(stop - start))
    #####################################################

if __name__ == "__main__":
    main()
