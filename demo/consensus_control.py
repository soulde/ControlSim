import numpy as np
import matplotlib.pyplot as plt
from control_env.implement.NonLinearForcedPendulum import NonLinearForcedPendulum
from control_env.implement.DistributedObserver import DistributedObserver
from control_env.cartographer.Cartographer import Graph
from tqdm import tqdm


def build_A_matrix(a_table):
    A = np.zeros([len(a_table), len(a_table)])

    for i, item in enumerate(a_table):
        for j in item:
            A[i, j] = 1
    return A


def build_D_matrix(a_table):
    D = np.zeros([len(a_table), len(a_table)])
    for i, item in enumerate(a_table):
        D[i, i] = len(item)
    return D


def build_B_matrix(link_table):
    B = np.zeros([len(link_table), len(link_table)])
    for item in link_table:
        B[item, item] = 1
    return B


def main():
    A_table = [[1, 2, 3], [4, 5], [6], [7], [9], [11], [8], [10], [], [], [], []]
    leader_link_table = [[0], [], [], [], [], [], [], [], [], [], [], []]
    A = build_A_matrix(A_table)
    D = build_D_matrix(A_table)
    L = D - A
    B = build_B_matrix(leader_link_table)

    leader = NonLinearForcedPendulum(np.array([-1, 2]))
    follower_list = []
    observer_list = []

    for i in range(len(A_table)):
        follower_list.append(NonLinearForcedPendulum(np.random.normal([0, 0], [1.57, 1])))
        observer_list.append(DistributedObserver(np.array([0, 0]), (L + B)[i], 6))

    K = np.array([[4]])
    u = np.zeros((len(A), 1))

    for i in tqdm(range(2000)):
        theta_d = leader(np.array([0]), 0.01)
        theta = np.array([item(u[i], 0.01) for i, item in enumerate(follower_list)])

        for j, item in zip(theta - theta_d, observer_list):
            item.observe(j)

        eit = np.array([item(u, 0.01) for i, item in enumerate(observer_list)])
        u = -(L @ theta + B @ (theta - theta_d)) @ K - eit @ K

    graph = Graph("state", 2)
    graph.plot_system_data(leader, 'leader')
    for i, item in enumerate(follower_list):
        graph.plot_system_data(item, 'follower ' + str(i))
    plt.show()


if __name__ == '__main__':
    main()
