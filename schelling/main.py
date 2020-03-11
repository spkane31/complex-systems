# Homework 2
# Due: March 31, 2020
# Sean Kane, Bayley King, Sean Rice

import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt
import datetime

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

        self.print_statements = True

        # Timeseries of happiness values
        self.happiness_ts = []

    def get_random_pt(self):
        return random.randint(0, self.N-1), random.randint(0, self.N-1)

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
            t_h = self.total_happiness()
            # Scenario where everyone is happy
            if t_h == self.population:
                break
            
            if self.print_statements: print(t_h / self.population)
            self.happiness_ts.append(t_h)

        self.space_to_image()

        # Produce timeseries for the happiness
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

    def social_network(self, n=5, p=3):
        # TODO: are the friends in same group (-1 or 1) or are they truly random

        # First find each agents "friends", randomly. This will be stored in a dictionary
        # this will require a lot of re-writing, but first thing that came to mind
        self.friends = {}
        
        # p = size of square neighborhood to look at
        self.p = p
        # n = number of friends
        self.n = n

        for i in range(self.N):
            for j in range(self.N):
                if self.space[i, j] != 0:
                    temp = []
                    while len(temp) < n:
                        x, y = self.get_random_pt()
                        if (x,y) not in temp:
                            temp.append((x, y))
                    self.friends[(i, j)] = temp
        print(len(self.friends))
        # 
        for _ in range(self.epochs):
            for i in range(self.N):
                for j in range(self.N):

                    if self.happiness(i, j) == 0 and self.space[i, j] != 0:
                        # Not "happy", look for new place
                        # print(self.space)

                        # print(i, j)

                        # print(self.friends[(i, j)])
                        locations = self.ask_friends(i, j)
                        # print(locations)
                        if len(locations) > 0:
                            new_loc = locations[random.randint(0, len(locations)-1)]
                        else:
                            x, y = self.get_random_pt()
                            while self.space[x, y] != 0:
                                x, y = self.get_random_pt()
                            new_loc = (x, y)
                        # print(new_loc)
                        self.friends[new_loc] = self.friends[(i, j)]
                        self.friends[(i, j)] = []

                        # print(self.friends[new_loc])
                        # print(self.friends[(i, j)])

                        self.space[new_loc[0], new_loc[1]] = self.space[i, j]
                        self.space[i, j] = 0

                        # print(self.space)

                        # quit()
            if self.print_statements: print(self.total_happiness() / self.population)
        
        self.space_to_image()
        pass

    def ask_friends(self, x, y):
        # TODO: the range to look at needs to be fixed
        f = self.friends[(x, y)]
        locs = []
        for friend in f:
            x, y = friend
            for i in range(-int(self.p/2), int((self.p+1)/2), 1):
                for j in range(-int(self.p/2), int((self.p+1)/2), 1):
                    x0, y0 = (x + i) % self.N, (y + j) % self.N
                    if self.space[x0, y0] == 0:
                        if (x0, y0) not in locs:
                            locs.append((x0, y0))

        return locs

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

        # Want the image to be 512 x 512
        scale = 512 / (self.N)

        img = Image.fromarray(im, 'RGB')
        img = img.resize((round(img.size[0]*scale), round(img.size[1]*scale)), Image.NEAREST)

        file_name = f"{datetime.datetime.now()}".split()[0]
        file_name += f"_k={self.k}_N={self.N}_epochs={self.epochs}"
        img.save(file_name+ ".png")
        # img.show()

if __name__ == "__main__":
    s = Schelling(N=100, k=4, epochs=100)
    print("Simulating...")
    s.social_network(n=5, p=4)
    print("Completed")