from scipy.integrate import solve_ivp
import numpy as np
from collections.abc import Callable
from abc import abstractmethod
from abc import ABC
import matplotlib.pyplot as plt

'''
@:param init_state state value when system start
@:param sys_fun system function F(x, u ,t) in equation dx/dt = F(x, u, t) or x(k) = F(x(k-1), u(k-1), k-1)
'''


class System(ABC):
    def __init__(self, name: str,
                 init_state: np.ndarray,
                 sim_period: float,
                 sys_fun: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray] = None,
                 out_fun: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray] = None
                 ):
        self.name = name
        self._state: np.ndarray = init_state
        self.state_names: list[str] = []
        self.sim_period: float = sim_period
        self.sys_fun: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray] = sys_fun
        self.out_fun: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray] = out_fun
        self._t_series: list = []
        self._x_series: list = []
        self.sys_time = 0

    @abstractmethod
    def __call__(self, u: np.ndarray = None, time: float = 0.1) -> np.ndarray:
        pass

    def __str__(self):
        return 'System: %s, state: %s' % (self.name, self._state)

    def set_state_name(self, *args):
        for i in args:
            self.state_names.append(i)

    def history(self):
        return np.hstack(self._t_series), np.hstack(self._x_series).T

    def plot(self, title=''):
        color_table = []
        t, x = self.history()
        print(x.shape)
        fig, axs = plt.subplots(x.shape[1], 1, figsize=[12.8, 8])
        fig.suptitle(title)
        while len(self.state_names) < x.shape[1]:
            self.state_names.append('')
        for i in range(x.shape[1]):
            axs[i].plot(t, x[:, i:i + 1], label=self.name)
            axs[i].set_xlabel('Time')
            axs[i].set_ylabel(self.state_names[i])
            axs[i].legend(bbox_to_anchor=(1.2, 1.0))
        fig.tight_layout()
        fig.align_labels()

        return axs


class ContinuousSystem(System):
    def __init__(self, name: str,
                 init_state: np.ndarray,
                 sys_fun: Callable[[np.ndarray, np.ndarray, float], np.ndarray] = None,
                 out_fun: Callable[[np.ndarray, np.ndarray, float], np.ndarray] = None,
                 sim_period: float = 0.0001):
        super().__init__(name, init_state, sim_period, sys_fun, out_fun)
        self._sys_time: float = 0

    def __call__(self, u: np.ndarray = None, time: float = 0.01) -> np.ndarray:
        assert self.sys_fun is not None

        ret = solve_ivp(lambda t, y: self.sys_fun(y, u, np.array([t])), [self._sys_time, self._sys_time + time],
                        self._state,
                        t_eval=np.linspace(self._sys_time, self._sys_time + time, int(1 / self.sim_period)))
        if ret.success:
            t_all = np.array(ret.t)
            x_all = np.array(ret.y)

            self._t_series.append(t_all[-1])
            self._x_series.append(x_all[:, -1:])

            self._state = x_all[:, -1]
            self._sys_time += time
        else:
            raise Exception('Simulate Error')
        return self.out_fun(self._state, u, np.array([self._sys_time]))


class DiscreteSystem(System):
    def __init__(self, name: str, init_state: np.ndarray,
                 sys_fun: Callable[[np.ndarray, np.ndarray, float], np.ndarray] = None,
                 out_fun: Callable[[np.ndarray, np.ndarray, float], np.ndarray] = None,
                 use_continuous_time=False,
                 time_per_count=0.01):
        super().__init__(name, init_state, time_per_count, sys_fun, out_fun)
        self._count = 0
        self._use_continuous_time = use_continuous_time

    def __call__(self, u: np.ndarray = None, time: float = 0.1) -> np.ndarray:
        assert self.sys_fun is not None
        t_all = []
        x_all = []
        for i in range(int(time)):
            self._state = self.sys_fun(self._state, u, np.array([self._count]))
            t_all.append(self._count * self.sim_period if self._use_continuous_time else self._count)
            x_all.append(self._state)
            self._count += 1

        self._t_series.append(np.array(t_all))
        self._x_series.append(np.array(x_all))

        return self.out_fun(self._state, u, np.array([self._count]))


class Source(System):
    def __init__(self, name: str, init_state: np.ndarray, sim_period=0.0001):
        super().__init__(name, init_state, sim_period=sim_period)

    def go(self, u: np.ndarray = None, time: float = 0.1) -> np.ndarray:
        pass

