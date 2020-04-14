# Bayley King
from os import system, name
from time import sleep
import timeit
import random
import numpy as np
import fit
from params import *
np.set_printoptions(precision=3)

def clear():
    _ = system('clear')

#clear()

class Particle():
    def __init__(self,currentLocation,name):
        self.name = name
        self.globalBest = randomInit()
        self.localBest = randomInit()
        self.selfBest = currentLocation
        self.currentPos = currentLocation
        self.currentVel = 0
        self.output = [self.globalBest, self.localBest, self.selfBest,self.currentPos]
        self.formattedList = []

    def printVals(self):
        self.formattedList = []
        for item in self.output:
            for i in item:
                #print(i)
                self.formattedList.append("%.2f"%i)        
        print(self.name)
        print('\tGlobal Best:',self.globalBest)
        #print('\tLocal Best:',self.localBest)
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
        print(self.name,'\tLocation:',self.currentPos,"\tFitness:{0:.3f}".format(fit.rastrigin(self.currentPos)))

  
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

def updateGlobal(newGlobal,swarm):
    for particle in swarm:
        particle.globalBest = newGlobal


def randomInit():
    x = np.random.random()*5.8
    y = np.random.random()*5.8
    return x,y

def printParts(swarm):
    [particle.printVals() for particle in swarm]


def main():
    # particle initialization
    for name in swarmNames:
        x,y = randomInit()
        swarm.append(Particle([x,y],name))
    
    # finds the best position from the init particle locations
    bestG = 9999
    for particle in swarm:
        if fit.rastrigin(particle.currentPos) < bestG:
           loc = particle.currentPos
           bestG = fit.rastrigin(particle.currentPos)
    updateGlobal(swarm,loc)

    printParts(swarm) # [particle.printVals() for particle in swarm] # prints particles


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
            if fit.rastrigin(particle.currentPos) < fit.rastrigin(particle.localBest):
                # Update global best
                particle.localBest = particle.currentPos
            # Part 4: Update velocity and position matrices
            update_velocity(particle)
            update_position(particle)
            #if i % 50== 0:
            #    print('Iterations:',i)            
            #    particle.printVals()

            #if i == iterations-1:
                #sleep(1)
            #    particle.printVals()

    stop = timeit.default_timer()
    print('##################################')
    bestVal = 9999
    for particle in swarm:
        #particle.printVals()
        particle.printPos()
        if fit.rastrigin(particle.currentPos) <= bestVal:
            bestVal = fit.rastrigin(particle.currentPos)
            best = particle
    print('W:',w,'\tC:',C,'\tS:',S)
    #best.printPos()
    print('Time to run: {0:.3f}'.format(stop - start))
    #####################################################

if __name__ == "__main__":
    main()