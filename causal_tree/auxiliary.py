import numpy as np
import matplotlib.pyplot as plt


def plot_figure1(width=8, height=5):
    def y(x1, x2):
        return (x1 < 0).astype(int) + (x1 >= 0).astype(int) * (
            2
            * (x2 < 0).astype(int)
            * (1 + (x2 > -2).astype(int) * (x1 > 2).astype(int))
            + 5 * (x2 >= 0).astype(int)
        )

    x = np.linspace(-5, 5, 100)
    X1, X2 = np.meshgrid(x, x)
    Y = y(X1, X2)
    fig = plt.figure()
    ax = plt.axes(projection="3d")

    ax.view_init(30, 230)
    ax.plot_surface(X1, X2, Y, rstride=1, cstride=1, cmap="viridis", edgecolor="none")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("y")
    ax.set_title("Figure 1")
    plt.rcParams["figure.figsize"] = [width, height]
    fig.show()
