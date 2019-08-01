import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# import matplotlib.pyplot as plt
from graphviz import Graph


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
    ax = plt.axes(projection=Axes3D.name)
    ax.view_init(30, 230)
    ax.plot_surface(X1, X2, Y, rstride=1, cstride=1, cmap="viridis", edgecolor="none")
    plt.xlabel("x1", fontsize=20)
    plt.ylabel("x2", fontsize=20)
    fig.suptitle("Figure 1", fontsize=30)
    plt.xticks(fontsize=16, rotation=90)
    plt.yticks(fontsize=16, rotation=90)
    # plt.zticks(fontsize=16, rotation=90)
    plt.rcParams["figure.figsize"] = [width, height]


def plot_figure2(ratio, width, height):
    dot = Graph(name="Figure 2")
    dot.attr(ratio=f"{ratio}", size=f"{width}, {height}!", label="Figure 2")
    # dot.node_attr.update(color='deepskyblue3', style='filled')
    dot.node("root", "x1 < 0")
    dot.node("lc", "0")
    dot.node("rc", "x2 < 0")
    dot.node("rlc", "5")
    dot.node("rrc", "x1 < 2")
    dot.node("rrlc", "2")
    dot.node("rrrc", "x2 < -2")
    dot.node("rrrlc", "4")
    dot.node("rrrrc", "0")
    dot.edge("root", "lc")
    dot.edge("root", "rc")
    dot.edge("rc", "rlc")
    dot.edge("rc", "rrc")
    dot.edge("rrc", "rrlc")
    dot.edge("rrc", "rrrc")
    dot.edge("rrrc", "rrrlc")
    dot.edge("rrrc", "rrrrc")
    return dot


def plot_figure3(width=8, height=5, view_x=60, view_y=230):
    def tau(x1, x2):
        tau1 = 0
        tau2 = 2 * (x1 < 0).astype(int) * (x2 > 0).astype(int)
        tau3 = 14 * (x1 < 0).astype(int) * (x2 < 0).astype(int)
        tau4 = -5 * (x1 > 0).astype(int) * (x2 < 0).astype(int)
        return tau1 + tau2 + tau3 + tau4

    x = np.linspace(-3, 3, 100)
    X1, X2 = np.meshgrid(x, x)
    T = tau(X1, X2)
    fig = plt.figure()
    ax = plt.axes(projection=Axes3D.name)
    ax.view_init(view_x, view_y)
    ax.plot_surface(X1, X2, T, rstride=1, cstride=1, cmap="viridis", edgecolor="none")
    plt.xlabel("x1", fontsize=20)
    plt.ylabel("x2", fontsize=20)
    fig.suptitle("Figure 3", fontsize=30)
    plt.xticks(fontsize=16, rotation=90)
    plt.yticks(fontsize=16, rotation=90)
    # plt.zticks(fontsize=16, rotation=90)
    plt.rcParams["figure.figsize"] = [width, height]


def plot_figure4(width=8, height=5, view_x=60, view_y=230, smoothness=20):
    def tau(x1, x2):
        tmp1 = 1 + (1 + np.exp(-smoothness * (x1 - 1 / 3))) ** (-1)
        tmp2 = 1 + (1 + np.exp(-smoothness * (x2 - 1 / 3))) ** (-1)
        return tmp1 * tmp2

    x = np.linspace(0, 1, 100)
    X1, X2 = np.meshgrid(x, x)
    T = tau(X1, X2)
    fig = plt.figure()
    ax = plt.axes(projection=Axes3D.name)
    ax.view_init(view_x, view_y)
    ax.plot_surface(X1, X2, T, rstride=1, cstride=1, cmap="viridis", edgecolor="none")
    plt.xlabel("x1", fontsize=20)
    plt.ylabel("x2", fontsize=20)
    fig.suptitle("Figure 4", fontsize=30)
    plt.xticks(fontsize=16, rotation=90)
    # plt.zticks(fontsize=16, rotation=90)
    plt.yticks(fontsize=16, rotation=90)
    plt.rcParams["figure.figsize"] = [width, height]
