# Homework 2
# Due: March 31, 2020
# Sean Kane, Bayley King, Sean Rice

import argparse
from collections import defaultdict
import datetime
import random
import time

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
        self.images = False

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
                happiness_temp.append(self.total_happiness()/self.population)

                for i in range(self.N):
                    for j in range(self.N):
                        x0, y0 = (x + i) % self.N, (y + j) % self.N
                        h = self.happiness(x0, y0)

                        if h == 0:
                            # If the agent is unhappy update that position
                            x1, y1 = self.find_random_open(x0, y0)
                            
                            self.space[x1, y1] = self.space[x0, y0]
                            self.space[x0, y0] = 0
                            self.open_spaces.remove((x1, y1))
                            self.open_spaces.append((x0, y0))

                t_h = self.total_happiness()
                # Scenario where everyone is happy
                if t_h == self.population:
                    # auto-fill happiness-over-epochs; won't change anymore
                    happiness_temp += [t_h/self.population] * (self.epochs - e)
                    # stop simulating this iteration
                    break
                
                if self.print_statements: print(t_h / self.population)
                happiness_temp.append(t_h/self.population)
            if self.images: self.space_to_image()
            # Produce timeseries for the happiness
            happiness_values.append(happiness_temp)
        temp = []
        for i in range(self.epochs):
            t = 0
            for j in range(self.iterations):
                t += happiness_values[j][i]
            temp.append(t / self.iterations)

        self.happiness_ts.append(temp)
        
        pass
    
    @staticmethod
    def count_neighbors_of_tribe(
        space: np.ndarray,
        x: int,
        y: int,
        tribe: int
    ) -> int:
        sx, sy = space.shape
        # 8-neighborhood with periodic boundary conditions
        neighbors = (
            ((x+ox) % sx, (y+oy) % sy)
            for ox in (-1,0,1) for oy in (-1,0,1)
            if ox != 0 or oy != 0 # don't include self
        )
        count = sum(1 if space[nx,ny]==tribe else 0 for (nx,ny) in neighbors)
        return count
    
    def happiness(self, x, y, tribe=None):
        tribe = self.space[x,y] if tribe is None else tribe
        n_same = self.count_neighbors_of_tribe(self.space, x, y, tribe)
        return 1 if n_same >= self.k else 0


    def __happiness_old(self, x, y):
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
            for _ in range(epochs):
                if self.print_statements: print(self.total_happiness() / self.population)
                happiness_temp.append(self.total_happiness()/self.population)

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
        for i in range(self.epochs):
            t = 0
            for j in range(self.iterations):
                t += happiness_values[j][i]
            temp.append(t / self.iterations)

        self.happiness_ts.append(temp)
        
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

    def sean_kane(self, distance=None, n=5, p=3):
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
            for _ in range(epochs):
                if self.print_statements: print(self.total_happiness() / self.population)
                happiness_temp.append(self.total_happiness()/self.population)

                for i in range(self.N):
                    for j in range(self.N):
                        # Here is where your algorithm goes, this iterates through the list in order from top left to bottom right so you may want to change that
                        # The 'for _ in range(epochs):' should stay, that makes through it goes through the same number of epochs each time

                        # TODO: insert code
                        if self.happiness(i, j) < 0.25 and self.space[i, j] != 0:
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
        for i in range(self.epochs):
            t = 0
            for j in range(self.iterations):
                t += happiness_values[j][i]
            temp.append(t / self.iterations)

        self.happiness_ts.append(temp)
        
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

    def sean_rice(self, prop_of_others: float=0.75):
        happiness_values = []
        # Sean Rice's choice policy
        for _ in range(self.iterations):
            # initial setup
            self.initialize_space()
            happiness_temp = [] # Stores the happiness values at each epoch for each iteration

            # get the list of nodes in a tribe (not spaces)
            nodes = [
                (i, j)
                for i in range(0, self.N) for j in range(0, self.N)
                if self.space[i, j] != 0
            ]
            # define a mapping from the node's "name" (original location) to
            # their current location. original *is* current location at init.
            node_locations = {node: node for node in nodes}
            # define a mapping from node tribe (-1 or 1) to a list of node
            # names and populate with nodes
            nodes_of_tribe = defaultdict(list)
            for node in nodes:
                x, y = node_locations[node]
                tribe = self.space[x, y]
                nodes_of_tribe[tribe].append(node)

            
            for _ in range(epochs):
                if self.print_statements: print(self.total_happiness() / self.population)
                happiness_temp.append(self.total_happiness()/self.population)

                random.shuffle(nodes)
                for node in nodes:
                    x, y = node_locations[node]
                    tribe = self.space[x,y]
                    # if already happy, do nothing.
                    if self.happiness(x, y) == 1:
                        continue

                    # otherwise, try to find someone to swap with
                    # start by picking a random subset of others
                    other_tribe = -1 * tribe
                    n_others = len(nodes_of_tribe[other_tribe])
                    others = random.sample(
                        nodes_of_tribe[other_tribe],
                        int(n_others * prop_of_others)
                    )
                    already_done = False
                    for other in others:
                        ox, oy = node_locations[other]
                        # would we be happy if we moved there?
                        we_would_swap = self.happiness(ox, oy, tribe=tribe) == 1
                        # would they be happy if they moved here?
                        other_would_swap = self.happiness(x, y, tribe=other_tribe) == 1
                        if we_would_swap and other_would_swap:
                            # we found someone to trade places with :)
                            # move us to them
                            self.space[ox, oy] = tribe
                            node_locations[node] = (ox, oy)
                            # move them to us
                            self.space[x, y] = other_tribe
                            node_locations[other] = (x, y)
                            already_done = True
                            break
                    if already_done:
                        continue

                    # last resort: random move.
                    # we are going to check spaces for one that makes us happy,
                    # but failing that we need to fall back to the spot that is
                    # *closest* to making us happy. we track that "best"
                    # space in these two variables and init it to our current.
                    spaces_to_check = random.sample(self.open_spaces, self.q)
                    best_space = node
                    best_fail_n = self.count_neighbors_of_tribe(self.space, x, y, tribe)
                    for space in spaces_to_check:
                        sx, sy = space
                        if self.happiness(sx, sy, tribe=tribe):
                            # we found an empty spot that makes us happy. swap.
                            best_space = space
                            break
                        else:
                            # otherwise, track it if it's an improvement to our current best
                            n_same = self.count_neighbors_of_tribe(self.space, sx, sy, tribe)
                            if n_same > best_fail_n:
                                best_fail_n = n_same
                                best_space = space
                    
                    self.space[sx, sy] = tribe
                    self.space[x, y] = 0
                    node_locations[node] = best_space
                    self.open_spaces.remove(best_space)
                    self.open_spaces.append((x, y))
                
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
        file_name = file_name.replace(":", ";")
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
    print(f"Simulating... (epochs: {epochs}, iterations: {iterations})")

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
        print(f"  Sean Kane")
        start = time.time()
        s.sean_kane()
        labels.append("Sean Kane")
        print(f"    Execution time: {round(time.time() - start, 2)} seconds")
    if args.run_king:
        print(f"  Bayley King")
        s.bayley_king()
        labels.append("Bayley King")
        print(f"    Execution time: {round(time.time() - start, 2)} seconds")
        
    if args.run_rice:
        print(f"  Sean Rice")
        s.sean_rice()
        labels.append("Sean Rice")
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
