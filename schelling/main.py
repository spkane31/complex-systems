# Homework 2
# Due: March 31, 2020
# Sean Kane, Bayley King, Sean Rice

import numpy as np
import random

class Schelling():

    def __init__(self, N=40, q=100):
        self.N = N
        self.population = int(N * N * 0.90)

        # each space can be -1, 0, 1
        self.space = np.zeros((N, N), dtype=int)

        # randomly initialize the locations
        for _ in range(int(self.population/2)):
            # first with +1's
            x, y = random.randint(0, self.N-1), random.randint(0, self.N-1)
            while self.space[x][y] != 0:
                x, y = random.randint(0, self.N-1), random.randint(0, self.N-1)
            self.space[x][y] = 1
            
        for _ in range(int(self.population/2)):
            # second with -1's
            x, y = random.randint(0, self.N-1), random.randint(0, self.N-1)
            while self.space[x][y] != 0:
                x, y = random.randint(0, self.N-1), random.randint(0, self.N-1)
            self.space[x][y] = -1

        # q is the number of empty sequential cells to move to
        self.q = q

        self.epochs = 50
        


    def random_move(self):
        # How to move each object in a random order? Could start at a different place each time

        # Find a random starting point and iterate from there
        x, y = random.randint(0, self.N-1), random.randint(0, self.N-1)

        for _ in range(self.epochs):
            for i in range(self.N):
                for j in range(self.N):
                    x0, y0 = (x + i) % self.N, (y + j) % self.N
                    print(self.space)
                    print(x0, y0)
                    h = self.rand_happiness(x0, y0)
                    print(h)
                    quit()


        pass

    def rand_happiness(self, x, y):
        # Calculate the happiness for a random location
        # Sums the values of all the neighbors
        total = -self.space[x, y]
        for i in range(-1, 2, 1):
            for j in range(-1, 2, 1):
                x0, y0 = (x + i) % self.N, (y + i) % self.N
                total += self.space[x0, y0]
                
        # Returns 1 if value greater than 0, -1 if less than 0, and 0 otherwise
        if total == 0: return 0
        return total/abs(total)

    def social_network(self):
        pass

    def sean_kane(self):
        # Sean Kane's choice policy
        pass

    def bayley_king(self):
        # Bayley's choice policy
        pass

    def sean_rice(self):
        # Sean Rice's choice policy
        pass

if __name__ == "__main__":
    s = Schelling(10)
    print("Simulating...")
    s.random_move()
    print("Completed")