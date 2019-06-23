import numpy as np


def simulate_treatment_data(agents=100, noise=False):
    x = np.random.normal(0, 1, size=2 * agents).reshape((agents, 2))
    treatment_status = np.random.binomial(1, 0.5, agents)
    treatment_index = np.array(treatment_status, dtype="bool")
    index0 = x[:, 0] > 0
    index1 = x[:, 1] > 0

    y = np.zeros(agents)

    index_quadrant2 = (~index0) * index1
    index_quadrant3 = (~index0) * (~index1)
    index_quadrant4 = index0 * (~index1)

    if not noise:
        y[index_quadrant2 * treatment_index] = 2
        y[index_quadrant2 * (~treatment_index)] = 0

        y[index_quadrant3 * treatment_index] = 20
        y[index_quadrant3 * (~treatment_index)] = 6

        y[index_quadrant4 * treatment_index] = -10
        y[index_quadrant4 * (~treatment_index)] = -5

    else:
        y[index_quadrant2 * treatment_index] = 2 + np.random.normal(
            0, 0.5, size=np.sum(index_quadrant2 * treatment_index)
        )
        y[index_quadrant2 * (~treatment_index)] = 0 + np.random.normal(
            0, 0.5, size=np.sum(index_quadrant2 * (~treatment_index))
        )
        y[index_quadrant3 * treatment_index] = 20 + np.random.normal(
            0, 0.5, size=np.sum(index_quadrant3 * treatment_index)
        )
        y[index_quadrant3 * (~treatment_index)] = 6 + np.random.normal(
            0, 0.5, size=np.sum(index_quadrant3 * (~treatment_index))
        )
        y[index_quadrant4 * treatment_index] = -10 + np.random.normal(
            0, 0.5, size=np.sum(index_quadrant4 * treatment_index)
        )
        y[index_quadrant4 * (~treatment_index)] = -5 + np.random.normal(
            0, 0.5, size=np.sum(index_quadrant4 * (~treatment_index))
        )

    return x, y, treatment_status
