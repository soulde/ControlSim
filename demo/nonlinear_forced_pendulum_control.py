import numpy as np
import matplotlib.pyplot as plt

from control_env.implement.NonLinearForcedPendulum import NonLinearForcedPendulum


def main():
    model = NonLinearForcedPendulum(np.array([-1, 2]))
    # K = np.array([1.7])
    u = np.array([0])
    for i in range(2000):
        ret = model(u, 0.02)
        print(model)
        # u = -K @ ret

    model.plot('NonLinear Forced Pendulum')
    plt.show()


if __name__ == '__main__':
    main()
