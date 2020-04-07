def rastrigin(x):
    import numpy as np
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