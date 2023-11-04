import random

import numpy as np
import matplotlib.pyplot as plt
from control_env.implement.AdaptiveController import LipschitzNonLinearSystem, AdaptiveController
from control_env.implement.DistributedObserver import DistributedObserver
from control_env.cartographer.Cartographer import Graph
from tqdm import tqdm

from control_env.implement import build_d_matrix, build_b_matrix, build_a_matrix


def init_c(dim):
    init = np.zeros([dim, dim])
    for i in range(dim):
        for j in range(dim):
            if i == j:
                init[i][j] = 0
            else:
                init[i][j] = (random.random() - 0.5)
    init += init.T
    return init


def main():
    A_table = [[1, 5], [0, 2, 3], [1, 5], [1, 4, 7], [3, 6], [0, 2], [7, 4], [3, 6]]
    gamma = np.array([[3.3676, 0.3935, -1.8917, 4.1236],
                      [0.3935, 0.046, -0.221, 0.4818],
                      [-1.8917, -0.221, 1.0627, -2.3164],
                      [4.1236, 0.4818, -2.3164, 5.0492]])
    F = np.array([[-1.8351, -0.2144, 1.0309, -2.247]])
    K = np.ones([len(A_table), len(A_table)])*0.1
    c_matrix = init_c(len(A_table))

    A = build_a_matrix(A_table)

    agents = []
    controllers = []
    init_randoms = np.array([[4.61340259, -3.13879007, 6.67698348, 5.96067365],
                             [-4.63540854, -2.09116592, 0.84450464, 4.72536001],
                             [8.89533042, 8.67666234, 3.96164427, 8.3698936],
                             [2.11970543, -4.7231706, 13.56099126, 7.64070929],
                             [6.93440844, -5.88378216, -2.48529715, -5.5078633],
                             [-0.8684402, -8.25981369, 3.24858731, -5.23096193],
                             [4.33688292, 2.20817975, -2.32717535, -1.18528629],
                             [3.38592995, -6.61097073, -1.02404912, 3.95437694]])
    for i in range(len(A_table)):
        # init_random = np.random.normal([0, 0, 0, 0], [5, 5, 5, 5]).T
        # print('agent{}:'.format(i), init_random)
        agents.append(LipschitzNonLinearSystem(np.array([[0, 1, 0, 0],
                                                         [-48.6, -1.25, 48.6, 0],
                                                         [0, 0, 0, 10],
                                                         [1.95, 0, -1.95, 0]]), np.array([[0, 21.6, 0, 0]]).T,
                                               lambda v: np.array([0, 0, 0, -0.33 * np.sin(v[2])]).T,
                                               init_randoms[i])
                      )

        controllers.append(AdaptiveController(
            gamma, F,
            np.array([j for index, j in enumerate(K[i]) if index in A_table[i]]),
            np.array([j for index, j in enumerate(c_matrix[i]) if index in A_table[i]])))

    u = np.zeros((len(A), 1))

    for _ in tqdm(range(1000)):
        x = np.array([item(u[i]) for i, item in enumerate(agents)]).T
        # print("x:", x)
        e = [np.array([x[:, i] - x[:, neighbor] for neighbor in A_table[i]]).T for i in range(len(A))]
        # print('e', e)
        u = np.array([item(e[i]) for i, item in enumerate(controllers)])
        # print('u', u)
        pass

    graph = Graph("state", 4)
    # graph.plot_system_data(leader, 'leader')
    for i, item in enumerate(agents):
        graph.plot_system_data(item, 'x' + str(i))
    # for index, item in enumerate(controllers):
    #     for i in item.state_names:
    #         # print(i)
    #         graph.plot_single_data(item, i, index)
    plt.show()


if __name__ == '__main__':
    main()
