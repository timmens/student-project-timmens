import numpy as np


def simulate_treatment_data(agents=100):
    x = np.random.normal(0, 1, size=2 * agents).reshape((agents, 2))
    treatment_status = np.random.binomial(1, 0.5, agents)
    treatment_index = np.array(treatment_status, dtype="bool")
    index0 = x[:, 0] > 0
    index1 = x[:, 1] > 0

    y = np.zeros(agents)

    #  index_quadrant1 = index0 * index1
    index_quadrant2 = (~index0) * index1
    index_quadrant3 = (~index0) * (~index1)
    index_quadrant4 = index0 * (~index1)

    y[index_quadrant2 * treatment_index] = 2  # treatment effect = 2
    y[index_quadrant2 * (~treatment_index)] = 0

    y[index_quadrant3 * treatment_index] = 2  # treatment effect = 1
    y[index_quadrant3 * (~treatment_index)] = 1

    y[index_quadrant4 * treatment_index] = -3  # treatment effect = 5
    y[index_quadrant4 * (~treatment_index)] = 2

    return x, treatment_status, y
