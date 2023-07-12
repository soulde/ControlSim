import numpy as np
import matplotlib.pyplot as plt
from control_env.Implement.InvertedPendulum import InvertedPendulum


def main():
    M = 2.
    m = 0.1
    l = 0.5
    K = np.array([[-545.54, -110.16, -380.39, -116.88]])
    x = np.array([0, 1, 0, 0])

    model = InvertedPendulum(M, m, l, x)
    for t in range(700):
        u = -K @ x
        x = model(u, 0.01)

    model.plot('Inverted Pendulum')
    # plt.plot(t, x)
    plt.show()


if __name__ == '__main__':
    main()
