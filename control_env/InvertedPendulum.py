import numpy as np
from control_env.constant import *
from control_env.System import ContinuousSystem


class InvertedPendulum(ContinuousSystem):
    def __init__(self, M, m, l, init_state, sim_period=0.0001):
        super().__init__(init_state, sim_period=sim_period)
        self.M = M
        self.m = m
        self.l = l
        self.sim_period = sim_period
        self.sys_time = 0

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
