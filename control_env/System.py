from scipy.integrate import solve_ivp
import numpy as np
from collections.abc import Callable
from abc import abstractmethod
from abc import ABC

'''
@:param init_state state value when system start
@:param sys_fun system function F(x, u ,t) in equation dx/dt = F(x, u, t) or x(k) = F(x(k-1), u(k-1), k-1)
'''


class System(ABC):
    def __init__(self, name: str,
                 init_state: np.ndarray,
                 sys_fun: Callable[[np.ndarray, np.ndarray, float], np.ndarray] = None,
                 out_fun: Callable[[np.ndarray, np.ndarray, float], np.ndarray] = None,
                 ):
        self.name = name
        self._state: np.ndarray = init_state
        self.sys_fun: Callable[[np.ndarray, np.ndarray, float], np.ndarray] = sys_fun
        self.out_fun: Callable[[np.ndarray, np.ndarray, float], np.ndarray] = out_fun
        self._t_series: list = []
        self._x_series: list = []

    @abstractmethod
    def go(self, u: np.ndarray = None, time: float = 0.1) -> np.ndarray:
        pass

    def history(self):
        return np.hstack(self._t_series), np.hstack(self._x_series).T


class ContinuousSystem(System):
    def __init__(self, name: str,
                 init_state: np.ndarray,
                 sys_fun: Callable[[np.ndarray, np.ndarray, float], np.ndarray] = None,
                 out_fun: Callable[[np.ndarray, np.ndarray, float], np.ndarray] = None,
                 sim_period: float = 0.0001):
        super().__init__(name, init_state, sys_fun, out_fun)
        self._sys_time: float = 0
        self._sim_period: float = sim_period

    def go(self, u: np.ndarray = None, time: float = 0.1) -> np.ndarray:
        assert self.sys_fun is not None

        ret = solve_ivp(lambda t, y: self.sys_fun(y, u, t), [self._sys_time, self._sys_time + time], self._state,
                        t_eval=np.linspace(self._sys_time, self._sys_time + time, int(1 / self._sim_period)))

        t_all = np.array(ret.t)
        x_all = np.array(ret.y)

        self._t_series.append(t_all)
        self._x_series.append(x_all)

        self._state = x_all[:, -1]
        self._sys_time += time

        return self.out_fun(self._state, u, self._sys_time)


class DiscreteSystem(System):
    def __init__(self, name: str, init_state: np.ndarray,
                 sys_fun: Callable[[np.ndarray, np.ndarray, float], np.ndarray] = None,
                 out_fun: Callable[[np.ndarray, np.ndarray, float], np.ndarray] = None,
                 use_continuous_time=False,
                 time_per_count=0.01):
        super().__init__(name, init_state, sys_fun, out_fun)
        self._count = 0
        self._time_per_count = time_per_count
        self._use_continuous_time = use_continuous_time

    def go(self, u: np.ndarray = None, time: float = 0.1) -> np.ndarray:
        assert self.sys_fun is not None
        t_all = []
        x_all = []
        for i in range(int(time)):
            self._state = self.sys_fun(self._state, u, self._count)
            t_all.append(self._count * self._time_per_count if self._use_continuous_time else self._count)
            x_all.append(self._state)
            self._count += 1

        self._t_series.append(np.array(t_all))
        self._x_series.append(np.array(x_all))

        return self.out_fun(self._state, u, self._count)


class Source(System):
    def __init__(self, name: str, init_state: np.ndarray):
        super().__init__(name, init_state)

    def go(self, u: np.ndarray = None, time: float = 0.1) -> np.ndarray:
        pass


class Environment:
    def __init__(self, sim_period: float = 0.0001):
        self.ele_list: list[str] = []
        self.elements: dict[str, System] = {}
        self.source: list[str] = []
        self.adjacency_table: dict[str, list] = {}
        self.sim_period = sim_period

    def register(self, element: System):
        self.ele_list.append(element.name)
        if isinstance(element, Source):
            self.source.append(element.name )
        self.elements[element.name] = element
        self.adjacency_table[element.name] = []

    def connect(self, out_element: System, in_element: System):
        self.adjacency_table[out_element.name].append(in_element.name)

    def simulate(self, t):
        pass

    def restart(self):
        pass
