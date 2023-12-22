import numpy as np
from control_env.core.constant import *
from control_env.core.System import ContinuousSystem, plt
from control_env.core import g


class InvertedPendulum(ContinuousSystem):
    def __init__(self, M, m, l_half, I, b, init_state, sim_period=0.001):
        super().__init__("InvertedPendulum", init_state, sim_period=sim_period)
        self.M = M
        self.m = m
        self.l_half = l_half
        self.I = I
        self.b = b
        self.sim_period = sim_period
        self.sys_time = 0
        self.set_state_name('position', 'velocity', 'angle', 'angular velocity')
        p = I * (M + m) + M * m * l_half ** 2
        self.A = np.array([[0, 1, 0, 0],
                           [0, -(I + m * l_half ** 2) * b / p, m ** 2 * g * l_half ** 2 / p, 0],
                           [0, 0, 0, 1],
                           [0, -m * l_half * b / p, m * g * l_half * (M + m) / p, 0]])
        self.B = np.array([[0],
                           [(I + m * l_half ** 2) / p],
                           [0],
                           [m * l_half / p]])
        self.C = np.eye(4)
        self.D = np.zeros([4, 1])

        self.sys_fun = lambda x, u, t: self.A @ self._state + self.B @ u
        self.out_fun = lambda x, u, t: self.C @ self._state + self.D @ u

    def plot(self, title=''):
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        color_table = []
        t, x = self.history()
        fig = plt.figure(figsize=[12.8, 8])

        fig.suptitle(title)
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], aspect='auto')
        # ax.ylim([0, 100])
        # ax.yticks(np.arange(0, 100, 10))
        while len(self.state_names) < x.shape[1]:
            self.state_names.append('')
        for i in range(x.shape[1]):
            ax.plot(t, x[:, i:i + 1], label=self.state_names[i])
        ax.set_xlabel('Time')
        ax.set_ylabel('value')
        ax.legend()

        # ax.align_labels()

        return ax
