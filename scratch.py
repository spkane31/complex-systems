# Bayley King
from os import system, name
from time import sleep
import timeit
import random
import numpy as np
np.set_printoptions(precision=3)

def clear():
    _ = system('clear')

w = 0.9 # Inertia
C =  0.5 # Cognitive Potential
S =  0.3 # Social Potential 
S2 = 0.2 # Local Social Potential

#clear()

class Particle():
    def __init__(self,currentLocation,name):
        self.name = name
        self.globalBest = randomInit()
        self.localBest = randomInit()
        self.selfBest = randomInit()
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
        print("Fitness:{0:.3f}".format(f(self.currentPos)))

  
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
    if particle.currentPos[0] >= 5.8:
        particle.currentPos[0] = 5.8
    elif particle.currentPos[0] <= -5.8:
        particle.currentPos[0] = -5.8
    if particle.currentPos[1] >= 5.8:
        particle.currentPos[1] = 5.8
    elif particle.currentPos[1] <= -5.8:
        particle.currentPos[1] = -5.8

def updateGlobal(newGlobal,swarm):
    for particle in swarm:
        particle.globalBest = newGlobal


def randomInit():
    x = np.random.random()*5.8
    y = np.random.random()*5.8
    return x,y

def f(pos):
    X = pos[0]
    Y = pos[1]
    Z = (X**2 - 10 * np.cos(2 * np.pi * X)) + \
          (Y**2 - 10 * np.cos(2 * np.pi * Y)) + 20
    return Z

swarmNames = ['P1','P2','P3','P4','P5','P6','P7','P8','P9','P10']
swarm = []

for name in swarmNames:
    x,y = randomInit()
    swarm.append(Particle([x,y],name))

start = timeit.default_timer()
iterations = 10000
for i in range(iterations):
    for particle in swarm:
        if i == 0:
            particle.printVals()
        # Part 1: If current position is less than the personal best,
        if f(particle.currentPos) < f(particle.selfBest):
            # Update personal best
            particle.selfBest = particle.currentPos
        # Part 2: If personal best is less than global best,
        if f(particle.currentPos) < f(particle.globalBest):
            updateGlobal(particle.currentPos,swarm)
            # Update global best
        # Part 3: If personal best is less than local best,
        '''
        if f(particle.currentPos) < f(particle.localBest):
            # Update global best
            particle.localBest = particle.currentPos
        '''
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
    particle.printVals()
    if f(particle.currentPos) <= bestVal:
        bestVal = f(particle.currentPos)
        best = particle
best.printVals()
print('Time to run: {0:.3f}'.format(stop - start))
