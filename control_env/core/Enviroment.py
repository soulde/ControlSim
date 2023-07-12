from System import System, Source
import numpy as np


class Environment:
    def __init__(self, sim_period: float = 0.0001):
        self.ele_list: list[System] = []
        self.time_scale_list: list[int] = []
        self.source: list[Source] = []
        self.adjacency_matrix: np.ndarray = np.array([])
        self.count = 0
        self.step = 0
        self.sim_period = sim_period
        self.activate_matrix: np.ndarray = np.array([])

    def register(self, element: System):
        assert (self.sim_period < element.sim_period)
        self.time_scale_list.append(int(element.sim_period // self.sim_period))
        if self.count == 0:
            self.adjacency_matrix = np.zeros([1, 1])
        else:
            self.adjacency_matrix = np.row_stack([self.adjacency_matrix, np.zeros([1, self.adjacency_matrix.shape[1]])])
            self.adjacency_matrix = np.column_stack(
                [self.adjacency_matrix, np.zeros([self.adjacency_matrix.shape[0], 1])])
        self.ele_list.append(element)
        self.count += 1
        if isinstance(element, Source):
            self.source.append(element)

    def connect(self, in_element: System, out_element: System):
        i = self.ele_list.index(in_element)
        j = self.ele_list.index(out_element)
        try:
            self.adjacency_matrix[i][j] = 1
        except IndexError:
            print("can not find selected elements")

    def simulate(self, t):
        for i in range(t / self.sim_period):
            self.one_step()
            self.step += 1

    def restart(self):
        pass

    def one_step(self):
        pass
