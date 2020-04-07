# Homework 2
# Due: March 31, 2020
# Sean Kane, Bayley King, Sean Rice

import argparse
from collections import defaultdict
import datetime
import random
import time
import statistics

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


class Schelling:
    def __init__(
        self,
        N=40,
        q=100,
        k=4,
        epochs=50,
        iterations=30,
        print_statements=False,
        save_images=False,
    ):
        self.N = N
        self.population = int(N * N * 0.90)

        # each space can be -1, 0, 1
        self.space = np.zeros((N, N), dtype=int)

        # q is the number of empty sequential cells to move to
        self.q = q
        # k is the number of neighbors to be around you to be happy
        self.k = k
        self.epochs = epochs

        self.print_statements = print_statements
        self.images = save_images

        # Timeseries of happiness values
        self.happiness_ts = []
        self.iterations = iterations

        self.open_spaces = []

    def initialize_space(self):
        self.space = np.zeros((self.N, self.N), dtype=int)
        # randomly initialize the locations
        for _ in range(int(self.population / 2)):
            # first with +1's
            x, y = random.randint(0, self.N - 1), random.randint(0, self.N - 1)
            while self.space[x][y] != 0:
                x, y = random.randint(0, self.N - 1), random.randint(0, self.N - 1)
            self.space[x][y] = 1

        for _ in range(int(self.population / 2)):
            # second with -1's
            x, y = random.randint(0, self.N - 1), random.randint(0, self.N - 1)
            while self.space[x][y] != 0:
                x, y = random.randint(0, self.N - 1), random.randint(0, self.N - 1)
            self.space[x][y] = -1

        self.open_spaces = list(map(tuple, np.argwhere(self.space == 0).tolist()))
        #self.open_spaces = [
        #    (i, j)
        #    for i in range(0, self.N)
        #    for j in range(0, self.N)
        #    if self.space[i, j] == 0
        #]

    def get_random_pt(self):
        return random.randint(0, self.N - 1), random.randint(0, self.N - 1)

    def random_move(self):
        POLICY_NAME = "random"
        happiness_values = []

        for _ in range(self.iterations):
            self.initialize_space()
            # How to move each object in a random order? Could start at a different place each time
            happiness_temp = []
            # Find a random starting point and iterate from there
            x, y = random.randint(0, self.N - 1), random.randint(0, self.N - 1)

            # initial conditions
            happiness_temp.append(self.total_happiness() / self.population)
            for e in range(self.epochs):
                valid_locations = [
                    (i, j)
                    for i in range(self.N)
                    for j in range(self.N)
                    if self.space[i, j] != 0
                ]
                random.shuffle(valid_locations)

                for location in valid_locations:
                    x0, y0 = location
                    h = self.happiness(x0, y0)
                    if h == 0:
                        # If the agent is unhappy update that position
                        x1, y1 = self.find_random_open(x0, y0)
                        self.space[x1, y1] = self.space[x0, y0]
                        self.space[x0, y0] = 0
                        self.open_spaces.remove((x1, y1))
                        self.open_spaces.append((x0, y0))

                t_h = self.total_happiness()
                happiness_temp.append(t_h / self.population)
                # Scenario where everyone is happy
                if t_h == self.population:
                    # we still need to produce a list of size (epochs+1)
                    while len(happiness_temp) != self.epochs + 1:
                         # lol this is awful but index math is hard
                        happiness_temp.append(t_h / self.population)
                    break # stop simulating this iteration
            
            # Produce timeseries for the happiness
            happiness_values.append(happiness_temp)

        epoch_stats = self.process_stats(happiness_values, self.iterations, self.epochs)
        self.happiness_ts.append(epoch_stats)
        self.complete_policy(POLICY_NAME, epoch_stats, True, self.images)

    @staticmethod
    def count_neighbors_of_tribe(space: np.ndarray, x: int, y: int, tribe: int) -> int:
        sx, sy = space.shape
        # 8-neighborhood with periodic boundary conditions
        neighbors = (
            ((x + ox) % sx, (y + oy) % sy)
            for ox in (-1, 0, 1)
            for oy in (-1, 0, 1)
            if ox != 0 or oy != 0  # don't include self
        )
        count = sum(1 if space[nx, ny] == tribe else 0 for (nx, ny) in neighbors)
        return count

    def happiness(self, x, y, tribe=None):
        tribe = self.space[x, y] if tribe is None else tribe
        if tribe == 0:
            return 0
        n_same = self.count_neighbors_of_tribe(self.space, x, y, tribe)
        return 1 if n_same >= self.k else 0

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
            h = self.happiness_value(
                self.open_spaces[i][0], self.open_spaces[i][1], self.space[x, y]
            )
            if h > cur_happiness:
                cur_happiness = h
                x0, y0 = self.open_spaces[i][0], self.open_spaces[i][1]
        # If none make it happier,
        if x0 == x and y0 == y:
            return self.open_spaces[random.randint(0, len(self.open_spaces) - 1)]

        return x0, y0

    def total_happiness(self):
        total = 0
        for i in range(self.N):
            for j in range(self.N):
                total += self.happiness(i, j)
        return total

    def social_network(self, n=5, p=3, epochs=100):
        POLICY_NAME = f"social-network-n={n}-p={p}"
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
                            if (x, y) not in temp:
                                temp.append((x, y))
                        self.friends[(i, j)] = temp

            happiness_temp = []

            for e in range(epochs):
                if self.print_statements:
                    print(self.total_happiness() / self.population)
                if e == 0:
                    happiness_temp.append(self.total_happiness() / self.population)

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
                                new_loc = locations[
                                    random.randint(0, len(locations) - 1)
                                ]
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
                if self.print_statements:
                    print(self.total_happiness() / self.population)
                happiness_temp.append(self.total_happiness() / self.population)

            happiness_values.append(happiness_temp)

        epoch_stats = self.process_stats(happiness_values, self.iterations, self.epochs)
        self.happiness_ts.append(epoch_stats)
        self.complete_policy(POLICY_NAME, epoch_stats, True, self.images)

    def ask_friends(self, x, y):
        # TODO: the range to look at needs to be fixed
        f = self.friends[(x, y)]
        locs = []
        for friend in f:
            x, y = friend
            for i in range(-int(self.p / 2), int((self.p + 1) / 2), 1):
                for j in range(-int(self.p / 2), int((self.p + 1) / 2), 1):
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
        POLICY_NAME = "sean-kane"
        self.n = n
        self.p = p

        if not distance:
            distance = int(self.N / 4)
        else:
            distance = min(distance, self.N / 4)

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
                            if (
                                (x, y) not in temp
                                and self.space[x, y] == self.space[i, j]
                                and self.euc_dist((i, j), (x, y)) < distance
                            ):
                                temp.append((x, y))
                        self.friends[(i, j)] = temp

            happiness_temp = (
                []
            )  # Stores the happiness values at each epoch for each iteration, the avg is taken care of later
            for e in range(epochs):
                if self.print_statements:
                    print(self.total_happiness() / self.population)
                if e == 0:
                    happiness_temp.append(self.total_happiness() / self.population)

                for i in range(self.N):
                    for j in range(self.N):
                        # Here is where your algorithm goes, this iterates through the list in order from top left to bottom right so you may want to change that
                        # The 'for _ in range(epochs):' should stay, that makes through it goes through the same number of epochs each time

                        # TODO: insert code
                        if self.happiness(i, j) < self.k and self.space[i, j] != 0:
                            locations = self.ask_friends(i, j)

                            if len(locations) > 0:
                                new_loc = locations[
                                    random.randint(0, len(locations) - 1)
                                ]
                                self.friends[new_loc] = self.friends[(i, j)]
                                self.friends[(i, j)] = []

                                self.space[new_loc[0], new_loc[1]] = self.space[i, j]
                                self.space[i, j] = 0
                            else:
                                # Do nothing if the friends can't find a better place
                                pass

                if self.print_statements:
                    print(self.total_happiness() / self.population)
                happiness_temp.append(self.total_happiness() / self.population)

            happiness_values.append(happiness_temp)
        
        epoch_stats = self.process_stats(happiness_values, self.iterations, self.epochs)
        self.happiness_ts.append(epoch_stats)
        self.complete_policy(POLICY_NAME, epoch_stats, True, self.images)


    def length_location(self,locations,i,j):
        dist = []
        for loc in locations:
            dist.append(((loc[0] - i)**2 + (loc[1] - j)**2)**.5)
        temp = list(zip(dist,locations))
        temp.sort()
        return temp[0][1]


    def bayley_king(self,n=5, p=3,epochs=10,randLocs=10):
        POLICY_NAME = "bayley-king"
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
            happiness_temp.append(self.total_happiness() / self.population)

            for _ in range(epochs):
                #print('Epoch:',e)
                for i in range(self.N):
                    for j in range(self.N):
                        # Here is where your algorithm goes, this iterates through the list in order from top left to bottom right so you may want to change that
                        # The 'for _ in range(epochs):' should stay, that makes through it goes through the same number of epochs each time

                        if self.happiness(i, j) == 0 and self.space[i, j] != 0: # if cell is unhappy and is an entity
 
                            locations = self.ask_friends(i, j)

                            if len(locations) > 0:
                                # calcualte the distance from current point to each suggested location
                                # sort by minnimum, go to minnimum travel location
                                new_loc = self.length_location(locations,i,j)
                            else:
                                # generate 10 random locations
                                # move to minnimum travel distance location
                                locations = []
                                for i in range(randLocs):
                                    locations.append(self.get_random_pt())
                                    while self.space[locations[i]] != 0:
                                        locations[i] = self.get_random_pt()
                                new_loc = self.length_location(locations,i,j)
                            # print(new_loc)
                            try:
                                self.friends[new_loc] = self.friends[(i, j)]
                            except:
                                print(new_loc)
                                print(i,j)
                            self.friends[(i, j)] = []

                            # print(self.friends[new_loc])
                            # print(self.friends[(i, j)])

                            self.space[new_loc[0], new_loc[1]] = self.space[i, j]
                            self.space[i, j] = 0

                            # print(self.space)

                            # quit()
                        
                        
                if self.print_statements:
                    print(self.total_happiness() / self.population)
                happiness_temp.append(self.total_happiness() / self.population)

            happiness_values.append(happiness_temp)

        epoch_stats = self.process_stats(happiness_values, self.iterations, self.epochs)
        self.happiness_ts.append(epoch_stats)
        self.complete_policy(POLICY_NAME, epoch_stats, True, self.images)


    def sean_rice(self, prop_of_others: float = 0.75):
        POLICY_NAME = "sean-rice"
        happiness_values = []
        # Sean Rice's choice policy
        for iteration in range(self.iterations):
            # initial setup
            self.initialize_space()
            happiness_temp = []
            happiness_temp.append(self.total_happiness() / self.population)

            # get the list of nodes in a tribe (not spaces)
            nodes = [
                (i, j)
                for i in range(0, self.N)
                for j in range(0, self.N)
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

            for epoch in range(epochs):
                random.shuffle(nodes)
                for node in nodes:
                    x, y = node_locations[node]
                    tribe = self.space[x, y]
                    # if already happy, do nothing.
                    if self.happiness(x, y) == 1:
                        continue

                    # otherwise, try to find someone to swap with
                    # start by picking a random subset of others
                    other_tribe = -1 * tribe
                    n_others = len(nodes_of_tribe[other_tribe])
                    others = random.sample(
                        nodes_of_tribe[other_tribe], int(n_others * prop_of_others)
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
                            if self.print_statements:
                                print(f"swap: {node}@{(x, y)} <-> {other}@{ox, oy}")
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
                            n_same = self.count_neighbors_of_tribe(
                                self.space, sx, sy, tribe
                            )
                            if n_same > best_fail_n:
                                best_fail_n = n_same
                                best_space = space
                    if best_space != node:
                        self.space[sx, sy] = tribe
                        self.space[x, y] = 0
                        node_locations[node] = best_space
                        self.open_spaces.remove(best_space)
                        self.open_spaces.append((x, y))

                if self.print_statements:
                    print(self.total_happiness() / self.population)
                happiness_temp.append(self.total_happiness() / self.population)

            happiness_values.append(happiness_temp)

        epoch_stats = self.process_stats(happiness_values, self.iterations, self.epochs)
        self.happiness_ts.append(epoch_stats)
        self.complete_policy(POLICY_NAME, epoch_stats, True, self.images)
    
    def complete_policy(self, policy_name, epoch_stats, print_std_dev, save_space_image):
        """
        General post-policy shared code to be run goes here.
        """
        print(policy_name)
        if print_std_dev == True:
            print(np.round(epoch_stats[1], decimals=3)) # print std devs
        if save_space_image == True:
            self.space_to_image(policy_name)

    @staticmethod
    def process_stats(happiness_data, iterations, epochs):
        """
        Calculates happiness statistics (average, std. dev.) over `iterations`
        number of trials for a time series of length `epochs`.

        happiness_data: A list (of length `iterations`) of lists (each of
        length `epochs`) containting all the happiness data at each epoch for
        each iteration. If the data isn't properly formed, raises `ValueError`.

        iterations: The number of iterations. Used to check the data shape.

        epochs: The number of data points per iteration. Used to check the data
        shape.

        Returns: A tuple of two lists `(means, stddevs)`, each of length
        `epochs`, corresponding to the means and std devs of the happiness data
        over the iterations for each epoch.
        """
        # first: *check everything*
        if len(happiness_data) != iterations:
            # fmt: off
            raise ValueError(f"expected outer list (iterations) length of {iterations}, got {len(happiness_data)}")
            # fmt: on
        epochs_len = epochs + 1
        for i, epoch_data in enumerate(happiness_data):
            if len(epoch_data) != epochs_len:
                # fmt: off
                raise ValueError(f"bad length of epoch timeseries; expected all {epochs_len},  mismatch at ({i}, {len(epoch_data)})")
                # fmt: on
        # okay! now we're sure we actually have a matrix
        # numpy is so much better for this stuff, we probably should just
        # directly be populating an ndarray rather than all that append business...
        data = np.array(happiness_data)  # shape (iterations, epochs)
        # axis 0 corresponds to iterations, which is what we want stats over.
        means = data.mean(axis=0)
        stddevs = data.std(axis=0, ddof=1)
        return (means, stddevs)

    def space_to_image(self, name_caption=None):
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

        img = Image.fromarray(im, "RGB")
        img = img.resize(
            (round(img.size[0] * scale), round(img.size[1] * scale)), Image.NEAREST
        )

        file_name = f"{datetime.datetime.now()}".split()[0]
        if name_caption is not None:
            file_name += f"_{name_caption}"
        file_name += f"_k={self.k}_N={self.N}_epochs={self.epochs}"
        file_name = file_name.replace(":", ";")
        img.save(file_name + ".png")

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
    ap.add_argument("-n", "--size", action="store", type=at_least(4), default=40, help="the size of the world (n x n) to simulate.")
    ap.add_argument("-e", "--epochs", action="store", type=at_least(1), default=30, help="the number of epochs to run. default: %(default)s")
    ap.add_argument("-i", "--iterations", action="store", type=at_least(3), default=10, help="the number of iterations to run. default: %(default)s")
    ap.add_argument("--print-statements", action="store_true", help="enable printing various logging/debug statements.")
    ap.add_argument("--save-images", action="store_true", help="enable saving images of the final space.")
    ap.add_argument("--show", action="store_true", help="show the final plot after program has run.")

    ap.add_argument("--all", action="store_true", required=False, dest="run_all", help="run all variants. overrides other choices. default: True if no other run options are given, else False")
    ap.add_argument("--random", action="store_true", required=False, dest="run_random", help="run the random policy. default: %(default)s")
    #ap.add_argument("--social", action="store_true", required=False, dest="run_social", help="run the social policy. default: %(default)s")
    #ap.add_argument("--kane", action="store_true", required=False, dest="run_kane",help="run sean kane's policy. default: %(default)s")
    ap.add_argument("--king", action="store_true", required=False, dest="run_king",help="run bayley king's policy. default: %(default)s")
    #ap.add_argument("--rice", action="store_true", required=False, dest="run_rice",help="run sean rice's policy. default: %(default)s")
    # fmt: on
    return ap


def process_arguments() -> argparse.Namespace:
    args = get_argparser().parse_args()

    # if no selections are made, run all by default
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


if __name__ == "__main__":
    args = process_arguments()

    epochs = args.epochs
    iterations = args.iterations
    labels = []

    s = Schelling(
        N=args.size,
        k=4,
        epochs=epochs,
        iterations=iterations,
        save_images=args.save_images,
        print_statements=args.print_statements,
    )
    print(f"Simulating (epochs: {epochs}, iterations: {iterations})... ")

    if args.run_random:
        print("  Random")
        start = time.time()
        s.random_move()
        labels.append("Random")
        print(f"    Execution time: {round(time.time() - start, 2)} seconds")
    '''
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
    '''
    if args.run_king:
        for n in [5, 10, 20]:
            for p in [3, 5, 7]:
                print(f"  Bayley King (n={n}, p={p})")
                start = time.time()
                s.bayley_king(n=n, p=p, epochs=epochs)
                labels.append(f"Bayley King (n={n}, p={p})")
                print(f"    Execution time: {round(time.time() - start, 2)} seconds")
    '''
    if args.run_rice:
        print(f"  Sean Rice")
        s.sean_rice()
        labels.append("Sean Rice")
        print(f"    Execution time: {round(time.time() - start, 2)} seconds")
    '''
    fig = plt.figure()
    ax = plt.subplot(111)
    
    x_axis = np.arange(1, epochs + 1 + 1)
    for (i, data) in enumerate(s.happiness_ts):
        if isinstance(data, tuple):
            # we have std dev info
            means, stddevs = data
            ax.errorbar(x_axis, means, yerr=stddevs, label=labels[i])
        else:
            ax.plot(x_axis, data, label=labels[i])

    plt.xlabel("Epochs")
    plt.ylabel("Happiness")
    chartBox = ax.get_position()
    ax.set_position([chartBox.x0, chartBox.y0, chartBox.width * 0.6, chartBox.height])
    ax.legend(loc="upper center", bbox_to_anchor=(1.45, 0.8), shadow=True, ncol=1)

    time_str = datetime.datetime.now().isoformat(sep="_").replace(":", ";")
    filename = f"{time_str}-timeseries-happiness.png"
    plt.savefig(filename)

    if args.show:
        plt.show()

    print("Completed")
