import numpy as np


def simulate_data(agents=100):
    x = np.random.normal(0, 1, size=2*agents).reshape((agents, 2))
    index0 = x[:, 0] > 0
    index1 = x[:, 1] > 0
    index = np.column_stack((index0, index1))

    y = np.empty(agents)
    y[np.apply_along_axis(all, 1, index)] = 0
    y[np.apply_along_axis(all, 1, ~index)] = 2
    y[np.apply_along_axis(lambda boo: boo[0] and not boo[1], 1, index)] = 1
    y[np.apply_along_axis(lambda boo: not boo[0] and boo[1], 1, index)] = 3

    return x, y
