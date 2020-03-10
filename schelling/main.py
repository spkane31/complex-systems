# Homework 2
# Due: March 31, 2020
# Sean Kane, Bayley King, Sean Rice

import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt

class Schelling():

    def __init__(self, N=40, q=100, k=4, epochs=50):
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
        # k is the number of neighbors to be around you to be happy
        self.k = k
        self.epochs = epochs

    def random_move(self):
        # How to move each object in a random order? Could start at a different place each time

        # Find a random starting point and iterate from there
        x, y = random.randint(0, self.N-1), random.randint(0, self.N-1)

        for _ in range(self.epochs):
            for i in range(self.N):
                for j in range(self.N):
                    x0, y0 = (x + i) % self.N, (y + j) % self.N
                    h = self.happiness(x0, y0)

                    if h == 0:
                        # Update that position
                        x1, y1 = self.find_random_open()
                        self.space[x1, y1] = self.space[x0, y0]
                        self.space[x0, y0] = 0
                
            # Scenario where everyone is happy
            if self.total_happiness() == self.population:
                break
            
            print(self.total_happiness() / self.population)
        self.space_to_image()
        pass

    def happiness(self, x, y):
        # Calculate the happiness for a random location

        # Sums the values of all the neighbors
        total = -self.space[x, y]
        for i in range(-1, 2, 1):
            for j in range(-1, 2, 1):
                x0, y0 = (x + i) % self.N, (y + j) % self.N
                total += self.space[x0, y0]
                
        # returns 1 if the cell is "happy", 0 otherwise
        if total >= self.k and self.space[x, y] == 1:
            return 1
        elif total <= -self.k and self.space[x, y] == -1:
            return 1
        return 0

    def find_random_open(self):
        x, y = random.randint(0, self.N-1), random.randint(0, self.N-1)
        while self.space[x, y] != 0:
            x, y = random.randint(0, self.N-1), random.randint(0, self.N-1)

        return x, y

    def total_happiness(self):
        total = 0
        for i in range(self.N):
            for j in range(self.N):
                total += self.happiness(i, j)
        return total

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

    def space_to_image(self):
        im = np.zeros((self.N, self.N, 3), dtype=np.uint8)
        for i in range(self.N):
            for j in range(self.N):
                if self.space[i, j] == 1:
                    im[i, j] = [255, 0, 0]
                elif self.space[i, j] == -1:
                    im[i, j] = [0, 0, 255]
                else:
                    im[i, j] = [255, 255, 255]
        # im = Image.fromarray(np.uint8(cm.gist_earth(im)) *255)
        # plt.imsave("image.png", im, cmap='Greys')
        # plt.imshow("image.png", im, interpolation='nearest')
        # plt.show()

        # Want the image to be 512 x 512
        scale = 1024 / (self.N)

        img = Image.fromarray(im, 'RGB')
        img = img.resize((round(img.size[0]*scale), round(img.size[1]*scale)))
        img.save('my.png')
        img.show()

if __name__ == "__main__":
    s = Schelling(N=100, k=4, epochs=100)
    print("Simulating...")
    s.random_move()
    print("Completed")