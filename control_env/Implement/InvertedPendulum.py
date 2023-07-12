import numpy as np
from control_env.Core.constant import *
from control_env.Core.System import ContinuousSystem, plt


class InvertedPendulum(ContinuousSystem):
    def __init__(self, M, m, l, init_state, sim_period=0.0001):
        super().__init__("InvertedPendulum", init_state, sim_period=sim_period)
        self.M = M
        self.m = m
        self.l = l
        self.sim_period = sim_period
        self.sys_time = 0
        self.set_state_name('position', 'velocity', 'angle', 'angular velocity')
        A = np.array([[0, 1, 0, 0],
                      [3 * (self.M + self.m) * g / (4 * self.M + self.m) / self.l, 0, 0, 0],
                      [0, 0, 0, 1],
                      [-3 * self.m * g / (4 * self.M + self.m), 0, 0, 0]])
        B = np.array([[0],
                      [-3 / (4 * self.M + self.m) / self.l],
                      [0],
                      [4 / (4 * self.M + self.m)]])
        C = np.eye(4)
        D = np.zeros([4, 1])

        self.sys_fun = lambda x, u, t: A @ self._state + B @ u
        self.out_fun = lambda x, u, t: C @ self._state + D @ u

    def plot(self, title=''):
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        color_table = []
        t, x = self.history()
        fig = plt.figure(figsize=[12.8, 8])
        fig.suptitle(title)
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], aspect=1)
        while len(self.state_names) < x.shape[1]:
            self.state_names.append('')
        for i in range(x.shape[1]):
            ax.plot(t, x[:, i:i + 1], label=self.state_names[i])
        ax.set_xlabel('Time')
        ax.set_ylabel('value')
        ax.legend()

        # ax.align_labels()

        return ax
