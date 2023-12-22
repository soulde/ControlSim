import numpy as np
import matplotlib.pyplot as plt
from control_env.implement.InvertedPendulum import InvertedPendulum
import tqdm


def main():
    M = 20000
    m = 30000
    l = 12.5
    I = 6250000
    b = 0.15
    K = np.array([[-0.0316, -56.4784, 9.8164e+05, 1.4600e+06]])
    x = np.array([0, 0, 0., 0])
    x_d = np.array([1000, 0, 0., 0])
    model = InvertedPendulum(M=M, m=m, l_half=l, I=I, b=b, init_state=x)
    sum_u = 0
    for t in tqdm.trange(150000):
        u = -K @ (x - x_d)
        sum_u += abs(u)
        # print(u)
        x = model(u, 0.1)
    # print(model.A, model.B, model.C, model.D)
    model.plot('Inverted Pendulum')
    # plt.plot(t, x)
    print(sum_u)
    plt.show()


if __name__ == '__main__':
    main()
