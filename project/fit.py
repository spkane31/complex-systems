def rastrigin(x):
    # https://en.wikipedia.org/wiki/Rastrigin_function
    import numpy as np

    ret = 0
    for xi in x:
        ret += (xi ** 2 - 10 * np.cos(2 * np.pi * xi))

    return ret + (10 * len(x))
    X = x[0]
    Y = x[1]
    Z = (X**2 - 10 * np.cos(2 * np.pi * X)) + \
        (Y**2 - 10 * np.cos(2 * np.pi * Y)) + 20
    return Z

def banana(x):
    x1 = x[0]
    x2 = x[1]
    return x1**4 - 2*x2*x1**2 + x2**2 + x1**2 - 2*x1 + 5

def con(x):
    x1 = x[0]
    x2 = x[1]
    return -(x1 + 0.25)**2 + 0.75*x2

def ackley(X):
    # http://benchmarkfcns.xyz/benchmarkfcns/ackleyfcn.html
    import numpy as np
    dims = len(X)
    a = 20
    b = 0.2
    c = 2*np.pi


    part1 = 0
    part2 = 0
    
    for xi in X:
        part1 += xi ** 2
        part2 += np.cos(c * xi)
    
    return -a * np.exp(-b * 1.0/dims * part1) - np.exp(1.0 / dims *part2) + a + np.exp(1)
    
def sphere(X):
    return sum([xi**2 for xi in X])

def styblinskiTang(X):
    if type(X) != list:
        return Exception("X must be a list type")
    # https://en.wikipedia.org/w/index.php?title=Styblinski%E2%80%93Tang_function&action=edit&redlink=1
    ret = 0
    for xi in X:
        ret += (xi ** 4 - 16 * xi**2 + 5 * xi)
    return ret / 2.0

# functions with (0, 0) have no bounds
bounds = {
    "rastrigin": (-5.12, 5.12),
    "banana": (0, 0),
    "con": (0, 0),
    "ackley": (-5, 5),
    "sphere": (0, 0),
    "styblinskiTang": (-5, 5)
}

string_to_func = {
    "rastrigin": rastrigin,
    "banana": banana,
    "con": con,
    "ackley": ackley,
    "sphere": sphere,
    "stblinski-tang": styblinskiTang
}

def actual_minimum(func, dims):
    if func == "rastrigin":
        return [0] * dims
    elif func == "banana":
        # TODO: validate this
        return [0] * dims
    elif func == "con":
        # TODO: validate this
        return [0] * dims
    elif func == "ackley":
        return [0] * dims
    elif func == "sphere":
        return [0] * dims
    elif func == "styblinskiTang":
        return [-2.903534] * dims
