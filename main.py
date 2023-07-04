import control as ctrl
import numpy as np
import matplotlib.pyplot as plt

#from control_env.InvertedPendulum import InvertedPendulum
def main():

    M = 2.
    m = 0.1
    g = 9.8
    l = 0.5
    A = np.array([[0, 1, 0, 0],
                  [3 * (M + m) * g / (4 * M + m) / l, 0, 0, 0],
                  [0, 0, 0, 1],
                  [-3 * m * g / (4 * M + m), 0, 0, 0]])
    B = np.array([[0],
                  [-3 / (4 * M + m) / l],
                  [0],
                  [4 / (4 * M + m)]])
    C = np.eye(4)
    D = np.zeros([4, 1])
    K = np.array([[-545.54, -110.16, -380.39, -116.88]])
    T = 100
    # print(A, B, C, D)
    sys = ctrl.ss(A, B, C, D)
    x0 = np.array([[0], [1], [0], [0]])

    t_series = []
    x_series = []
    for t in range(700):
        if t == 0:
            x = x0
        u = -K @ x
        time_stamp = np.linspace(t, t + 1, 1000) / T
        t_all, x_all = ctrl.forced_response(sys, time_stamp, np.ones([1, 1000]) * u, x)
        x = x_all[:, -2:-1]
        t_series.append(t_all)
        x_series.append(x_all)
        # print(time_stamp)
    t_array = np.hstack(t_series).reshape([1, -1]).T
    x_array = np.hstack(x_series).T
    plt.plot(t_array, x_array)
    plt.show()

if __name__ == '__main__':
    main()
