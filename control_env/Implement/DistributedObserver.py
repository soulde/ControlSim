import numpy as np
from control_env.Core.Observer import Observer


class DistributedObserver(Observer):
    def __init__(self, init_state, L_plus_B_matrix, l, sim_period=0.0001):
        super().__init__(init_state, "DistributedObserver", sim_period=sim_period)
        self.set_state_name('position', 'velocity')

        def _sys_fun(x, u, t):
            return np.hstack([x[1] - l * (x[0] - self._obs), L_plus_B_matrix @ u - l * l * (x[0] - self._obs)])

        self.sys_fun = _sys_fun
        self.out_fun = lambda x, u, t: x[1:]
