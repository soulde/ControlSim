import numpy as np
from control_env.core.System import ContinuousSystem
from abc import ABC


class Observer(ContinuousSystem, ABC):
    def __init__(self, init_state, name='Observer', sim_period=0.0001):
        super().__init__(name, init_state, sim_period=sim_period)
        self._obs: np.ndarray = None

    def observe(self, obs):
        self._obs = obs
