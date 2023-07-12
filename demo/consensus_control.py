import numpy as np
import matplotlib.pyplot as plt

from control_env.implement.NonLinearForcedPendulum import NonLinearForcedPendulum
from control_env.implement.DistributedObserver import DistributedObserver


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
    print(L, B, A)
    leader = NonLinearForcedPendulum(np.array([-1, 2]))
    follower_list = []
    observer_list = []

    for i in range(len(A_table)):
        follower_list.append(NonLinearForcedPendulum(np.random.normal([0, 0], [1.57, 1])))
        observer_list.append(DistributedObserver(np.array([0, 0]), (L + B)[i], 6))
        # print((L + B)[i])
    K = np.array([[4]])
    u = np.zeros((len(A), 1))

    for i in range(1000):
        theta_d = leader(np.array([0]), 0.01)
        # print(u.shape)
        theta = np.array([item(u[i], 0.01) for i, item in enumerate(follower_list)])

        for j, item in zip(theta, observer_list):
            item.observe(j)

        estimated = np.array(
            [item(u, 0.01) for i, item in enumerate(observer_list)])
        # print(theta.shape, estimated.shape, )
        u = -(L + B) @ theta @ K - estimated @ K
        # print(u)
        # print(u.shape)
    leader.plot()
    for i, item in enumerate(follower_list):
        item.plot('follower_' + str(i))
    plt.show()


if __name__ == '__main__':
    main()
