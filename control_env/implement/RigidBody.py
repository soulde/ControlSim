import numpy as np
from control_env.core.System import ContinuousSystem


class RigidBody(ContinuousSystem):
    def __init__(self, init_state, sim_period=0.0001):
        super().__init__("RigidBody", init_state, sim_period=sim_period)
        self.set_state_name('w', 'x', 'y', 'z')
        self.sys_fun = lambda x, u, t: 0.5 * (np.array([[0, -u[0], -u[1], -u[2]],
                                                        [u[0], 0, -u[2], -u[1]],
                                                        [u[1], -u[2], 0, u[1]],
                                                        [u[2], u[1], -u[0], 0]])) @ x
        self.out_fun = lambda x, u, t: x
