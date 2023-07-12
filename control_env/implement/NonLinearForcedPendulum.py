import numpy as np
from control_env.core.System import ContinuousSystem


class NonLinearForcedPendulum(ContinuousSystem):
    def __init__(self, init_state, sim_period=0.0001):
        super().__init__("NonLinearForcedPendulum", init_state, sim_period=sim_period)
        self.set_state_name('position', 'velocity')
        self.sys_fun = lambda x, u, t: np.hstack([x[1], -np.sin(x[0:1]) - 0.25 * x[1:] + 1.5 * np.cos(2.5 * t) + u])
        self.out_fun = lambda x, u, t: x[0:1]
