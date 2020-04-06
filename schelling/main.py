# Homework 2
# Due: March 31, 2020
# Sean Kane, Bayley King, Sean Rice

import argparse
import datetime
import random
import time
import statistics

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

class Schelling():

    def __init__(self, N=40, q=100, k=4, epochs=50, iterations=30):
        self.N = N
        self.population = int(N * N * 0.90)

        # each space can be -1, 0, 1
        self.space = np.zeros((N, N), dtype=int)

        # q is the number of empty sequential cells to move to
        self.q = q
        # k is the number of neighbors to be around you to be happy
        self.k = k
        self.epochs = epochs

        self.print_statements = False
        self.images = True

        # Timeseries of happiness values
        self.happiness_ts = []
        self.iterations = iterations

        self.open_spaces = []

    def initialize_space(self):
        self.space = np.zeros((self.N, self.N), dtype=int)
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

        for i in range(self.N):
            for j in range(self.N):
                if self.space[i, j] == 0:
                    self.open_spaces.append((i, j))

    def get_random_pt(self):
        return random.randint(0, self.N-1), random.randint(0, self.N-1)

    def random_move(self):
        happiness_values = []

        for _ in range(self.iterations):
            self.initialize_space()
            # How to move each object in a random order? Could start at a different place each time
            happiness_temp = []
            # Find a random starting point and iterate from there
            x, y = random.randint(0, self.N-1), random.randint(0, self.N-1)

            for e in range(self.epochs):
                if self.print_statements: print(self.total_happiness() / self.population)
                if e == 0: happiness_temp.append(self.total_happiness()/self.population)

                valid_locations = []
                for i in range(self.N):
                    for j in range(self.N):
                        if self.space[i, j] != 0:
                            valid_locations.append((i, j))
                random.shuffle(valid_locations)

                for location in valid_locations:
                    x0, y0 = location
                    h= self.happiness(x0, y0)

                    if h == 0:
                        x1, y1 = self.find_random_open(x0, y0)
                            
                        self.space[x1, y1] = self.space[x0, y0]
                        self.space[x0, y0] = 0
                        self.open_spaces.remove((x1, y1))
                        self.open_spaces.append((x0, y0))

                t_h = self.total_happiness()
                # Scenario where everyone is happy
                if t_h == self.population:
                    break
                
                if self.print_statements: print(t_h / self.population)
                happiness_temp.append(t_h/self.population)
            if self.images: self.space_to_image()
            # Produce timeseries for the happiness
            happiness_values.append(happiness_temp)
        temp = []
        stddev = []
        for i in range(self.epochs):
            t = 0
            vals = []
            for j in range(self.iterations):
                t += happiness_values[j][i]
                vals.append(happiness_values[j][i])
            temp.append(t / self.iterations)
            # stddev.append(statistics.stdev(vals))

        self.happiness_ts.append(temp)
        print(f"Random move")
        for s in stddev: print(round(s, 3))
        pass

    def happiness(self, x, y):
        # Calculate whether an agent is happy at it's own location

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

    def happiness_value(self, x, y, cur_value):
        # Calculate the happiness for a random location
        total = 0
        # Sums the values of all the neighbors if they're the same as the cur_value
        for i in range(-1, 2, 1):
            for j in range(-1, 2, 1):
                x0, y0 = (x + i) % self.N, (y + j) % self.N
                if self.space[x0, y0] == cur_value:
                    total += self.space[x0, y0]

        # Looks at 8 cells, perfect happiness is all similar
        return total / 8

    def find_random_open(self, x, y):
        cur_happiness = self.happiness_value(x, y, self.space[x, y])

        x0, y0 = x, y
        for i in range(min(len(self.open_spaces), self.q)):
            h = self.happiness_value(self.open_spaces[i][0], self.open_spaces[i][1], self.space[x, y])
            if h > cur_happiness:
                cur_happiness = h
                x0, y0 = self.open_spaces[i][0], self.open_spaces[i][1]
        # If none make it happier, 
        if x0 == x and y0 == y:
            return self.open_spaces[random.randint(0, len(self.open_spaces)-1)]
            
        return x0, y0

    def total_happiness(self):
        total = 0
        for i in range(self.N):
            for j in range(self.N):
                total += self.happiness(i, j)
        return total

    def social_network(self, n=5, p=3, epochs=100):
        happiness_values = []
        
        # p = size of square neighborhood to look at
        self.p = p
        # n = number of friends
        self.n = n

        for _ in range(self.iterations):
            self.initialize_space()
            
            # First find each agents "friends", randomly. This will be stored in a dictionary
            # this will require a lot of re-writing, but first thing that came to mind
            self.friends = {}

            for i in range(self.N):
                for j in range(self.N):
                    if self.space[i, j] != 0:
                        temp = []
                        while len(temp) < n:
                            x, y = self.get_random_pt()
                            if (x,y) not in temp:
                                temp.append((x, y))
                        self.friends[(i, j)] = temp
            
            happiness_temp = []
            for e in range(epochs):
                if self.print_statements: print(self.total_happiness() / self.population)
                if e == 0: happiness_temp.append(self.total_happiness()/self.population)

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
                happiness_temp.append(self.total_happiness()/self.population)
            
            happiness_values.append(happiness_temp)
            if self.images: self.space_to_image()

        temp = []
        stddev = []
        for i in range(self.epochs):
            t = 0
            vals = []
            for j in range(self.iterations):
                t += happiness_values[j][i]
                vals.append(happiness_values[j][i])
            temp.append(t / self.iterations)
            stddev.append(statistics.stdev(vals))

        self.happiness_ts.append(temp)
        print(f"social_policy n={n} p={p}")
        for s in stddev: print(round(s, 3))
        if self.images: self.space_to_image()
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
                        h = self.happiness_value(x0, y0, self.space[x, y])
                        if h > 0 and self.space[x, y] > 0:
                            locs.append((x0, y0))
                        if h < 0 and self.space[x, y] < 0:
                            locs.append((x0, y0))

        return locs

    def euc_dist(self, a, b):
        ax, ay = a
        bx, by = b
        return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5

    def sean_kane(self, distance=None, n=5, p=3, epochs=30):
        self.n = n
        self.p = p
        
        if not distance:
            distance = int(self.N/4)
        else:
            distance = min(distance, self.N/4)

        happiness_values = []
        # Sean Kane's choice policy
        for _ in range(self.iterations):
            # Start by creating a new starting space
            self.initialize_space()

            # Find "friends" randomly, except they have to be the same "type" and less than or equal
            # to distance from each other. Each agent has 5 friends.
            self.friends = {}
            for i in range(self.N):
                for j in range(self.N):
                    if self.space[i, j] != 0:
                        temp = []
                        while len(temp) < n:
                            x, y = self.get_random_pt()
                            if (x, y) not in temp and self.space[x, y] == self.space[i, j] and self.euc_dist((i, j), (x, y)) < distance:
                                temp.append((x, y))
                        self.friends[(i, j)] = temp

            

            happiness_temp = [] # Stores the happiness values at each epoch for each iteration, the avg is taken care of later
            for e in range(epochs):
                if self.print_statements: print(self.total_happiness() / self.population)
                if e == 0: happiness_temp.append(self.total_happiness()/self.population)

                for i in range(self.N):
                    for j in range(self.N):
                        # Here is where your algorithm goes, this iterates through the list in order from top left to bottom right so you may want to change that
                        # The 'for _ in range(epochs):' should stay, that makes through it goes through the same number of epochs each time

                        # TODO: insert code
                        if self.happiness(i, j) < self.k and self.space[i, j] != 0:
                            locations = self.ask_friends(i, j)

                            if len(locations) > 0:
                                new_loc = locations[random.randint(0, len(locations)-1)]
                                self.friends[new_loc] = self.friends[(i, j)]
                                self.friends[(i, j)] = []

                                self.space[new_loc[0], new_loc[1]] = self.space[i, j]
                                self.space[i, j] = 0
                            else:
                                # Do nothing if the friends can't find a better place
                                pass
            
                if self.print_statements: print(self.total_happiness() / self.population)
                happiness_temp.append(self.total_happiness()/self.population)
                
            happiness_values.append(happiness_temp)
            # Save the image of the final neighborhood if this switch is on
            if self.images: self.space_to_image()

        # This goes through calculating the average happiness at each epoch, leave this alone.
        temp = []
        stddev = []
        for i in range(self.epochs):
            t = 0
            vals = []
            for j in range(self.iterations):
                t += happiness_values[j][i]
                vals.append(happiness_values[j][i])
            temp.append(t / self.iterations)
            # stddev.append(statistics.stdev(vals))

        self.happiness_ts.append(temp)
        print(f"kane")
        for s in stddev: print(round(s, 3))
        
        if self.images: self.space_to_image()

    def bayley_king(self):
        happiness_values = []
        # Bayley's choice policy
        for _ in range(self.iterations):
            # Start by creating a new starting space
            self.initialize_space()
            

            happiness_temp = [] # Stores the happiness values at each epoch for each iteration, the avg is taken care of later
            for _ in range(epochs):
                if self.print_statements: print(self.total_happiness() / self.population)
                happiness_temp.append(self.total_happiness()/self.population)

                for i in range(self.N):
                    for j in range(self.N):
                        # Here is where your algorithm goes, this iterates through the list in order from top left to bottom right so you may want to change that
                        # The 'for _ in range(epochs):' should stay, that makes through it goes through the same number of epochs each time

                        # TODO: insert code

                        pass

                        
                        
                if self.print_statements: print(self.total_happiness() / self.population)
                happiness_temp.append(self.total_happiness()/self.population)
            
            happiness_values.append(happiness_temp)
            # Save the image of the final neighborhood if this switch is on
            if self.images: self.space_to_image()

        # This goes through calculating the average happiness at each epoch, leave this alone.
        temp = []
        for i in range(self.epochs):
            t = 0
            for j in range(self.iterations):
                t += happiness_values[j][i]
            temp.append(t / self.iterations)

        self.happiness_ts.append(temp)
        
        if self.images: self.space_to_image()

    def sean_rice(self):
        raise NotImplementedError("i haven't done this yet")
        happiness_values = []
        # Sean Rice's choice policy
        for _ in range(self.iterations):
            # Start by creating a new starting space
            self.initialize_space()
            

            happiness_temp = [] # Stores the happiness values at each epoch for each iteration, the avg is taken care of later
            for _ in range(epochs):
                if self.print_statements: print(self.total_happiness() / self.population)
                happiness_temp.append(self.total_happiness()/self.population)

                for i in range(self.N):
                    for j in range(self.N):
                        # Here is where your algorithm goes, this iterates through the list in order from top left to bottom right so you may want to change that
                        # The 'for _ in range(epochs):' should stay, that makes through it goes through the same number of epochs each time

                        # TODO: insert code

                        pass
                        
                if self.print_statements: print(self.total_happiness() / self.population)
                happiness_temp.append(self.total_happiness()/self.population)
            
            happiness_values.append(happiness_temp)
            # Save the image of the final neighborhood if this switch is on
            if self.images: self.space_to_image()

        # This goes through calculating the average happiness at each epoch, leave this alone.
        temp = []
        for i in range(self.epochs):
            t = 0
            for j in range(self.iterations):
                t += happiness_values[j][i]
            temp.append(t / self.iterations)

        self.happiness_ts.append(temp)
        
        if self.images: self.space_to_image()

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

def get_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--epochs", action="store", type=int, default=30, help="the number of epochs to run. default: %(default)s")
    ap.add_argument("-i", "--iterations", action="store", type=int, default=32, help="the number of iterations to run. default: %(default)s")
    ap.add_argument("--show", action="store", type=bool, default=False, help="show the final plot after program has run.")

    ap.add_argument("--all", action="store_true", required=False, dest="run_all", help="run all variants. overrides other choices. default: True if no other run options are given, else False")
    ap.add_argument("--random", action="store_true", required=False, dest="run_random", help="run the random policy. default: %(default)s")
    ap.add_argument("--social", action="store_true", required=False, dest="run_social", help="run the social policy. default: %(default)s")
    ap.add_argument("--kane", action="store_true", required=False, dest="run_kane",help="run sean kane's policy. default: %(default)s")
    ap.add_argument("--king", action="store_true", required=False, dest="run_king",help="run bayley king's policy. default: %(default)s")
    ap.add_argument("--rice", action="store_true", required=False, dest="run_rice",help="run sean rice's policy. default: %(default)s")
    return ap

def process_arguments() -> argparse.Namespace:
    args = get_argparser().parse_args()

    # if no selections are made, run all by default
    none_selected = not any(
        [arg_setting
        for arg_name, arg_setting in vars(args).items()
        if arg_name.startswith("run_")]
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
    labels = []

    s = Schelling(N=40, k=4, epochs=epochs, iterations=iterations)
    print("Simulating...")

    if args.run_random:
        print("  Random")
        start = time.time()
        s.random_move()
        labels.append("Random")
        print(f"    Execution time: {round(time.time() - start, 2)} seconds")
    
    if args.run_social:
        for n in [5, 10, 20]:
            for p in [3, 5, 7]:
                print(f"  Social (n={n}, p={p})")
                start = time.time()
                s.social_network(n=n, p=p, epochs=epochs)
                labels.append(f"Social (n={n}, p={p})")
                print(f"    Execution time: {round(time.time() - start, 2)} seconds")

    if args.run_kane:
        for n in [5, 10, 20]:
            for p in [3, 5, 7]:
                print(f"  Sean Kane (n={n}, p={p})")
                start = time.time()
                s.sean_kane(n=n, p=p, epochs=epochs)
                labels.append(f"Sean Kane (n={n}, p={p})")
                print(f"    Execution time: {round(time.time() - start, 2)} seconds")

    if args.run_king:
        print(f"  Bayley King")
        s.bayley_king()
        labels.append("Bayley King")
        print(f"    Execution time: {round(time.time() - start, 2)} seconds")
        
    if args.run_rice:
        print(f"  Sean Rice")
        try:
            s.sean_rice()
            labels.append("Sean Rice")
        except NotImplementedError: 
            # nobody cares
            pass
        print(f"    Execution time: {round(time.time() - start, 2)} seconds")

    fig = plt.figure()
    ax = plt.subplot(111)

    x_axis = list(range(1, epochs+1))
    for (i, h) in enumerate(s.happiness_ts):
        ax.plot(x_axis, h, label=labels[i])
    
    plt.xlabel('Epochs')
    plt.ylabel('Happiness')
    chartBox = ax.get_position()
    ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.6, chartBox.height])
    ax.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=True, ncol=1)
    
    time_str = datetime.datetime.now().isoformat(sep="_").replace(":", ";")
    filename = f"{time_str}-timeseries-happiness.png"
    plt.savefig(filename)

    if args.show:
        plt.show()

    print("Completed")