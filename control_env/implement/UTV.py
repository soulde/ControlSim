import math

import numpy as np
from control_env.core.System import ContinuousSystem


class UTVDynamic(ContinuousSystem):
    def __init__(self, init_state, utv_type='Omni', sim_period=0.0001):
        super().__init__("DistributedObserver", init_state=init_state, sim_period=sim_period)
        self.set_state_name('position')
        if utv_type == 'Omni':
            self.sys_fun = lambda x, u, t: np.hstack([u])
            self.out_fun = lambda x, u, t: x[1:]
        elif utv_type == 'Ackerman':
            self.sys_fun = lambda x, u, t: np.hstack([u[0] * math.sin(x[2]), u[1] * math.sin(x[2]), u[1]])
            self.out_fun = lambda x, u, t: x[1:]
