import numpy as np
from control_env.core.System import ContinuousSystem


class LipschitzNonLinearSystem(ContinuousSystem):
    def __init__(self, matrix_a, matrix_b, nonlinear_f, init_state, sim_period=0.001):
        super().__init__("LipschitzNonLinearSystem", init_state, sim_period=sim_period)
        self.set_state_name(
            *["x" + str(i) for i in range(init_state.shape[0])]
        )

        self.sys_fun = lambda x, u, t: matrix_a @ x + matrix_b @ u + nonlinear_f(x)
        self.out_fun = lambda x, u, t: x


class AdaptiveController(ContinuousSystem):
    def __init__(self, gamma, F, k_i, init_state, sim_period=0.001):
        super().__init__("LipschitzNonLinearSystem", init_state, sim_period=sim_period)
        self.set_state_name(*[
            "c" + str(i) for i in range(init_state.shape[0])
        ])

        def sys_debug(x, u, t):
            # print('u:', u[:, 1:].T@gamma@u[:, 1:])
            # print('di:', u.T @ gamma @ u)
            # print('dc:',k_i * np.diagonal(u.T @ gamma @ u))
            # print(np.diagonal(u.T @ gamma @ u))
            return k_i * np.diagonal(u.T @ gamma @ u)

        self.sys_fun = lambda x, u, t: k_i * np.diagonal(u.T @ gamma @ u)
        # self.sys_fun = sys_debug

        def out_debug(x, u, t):
            # print('c:', x)
            return F @ u @ x.T

        self.out_fun = lambda x, u, t: F @ u @ x.T
        # self.out_fun = out_debug
