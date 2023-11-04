import numpy as np


class Field:
    def __init__(self, size):
        self.grid = np.zeros(size, dtype=float)

    def random(self):
        pass

    def check_collision(self, pos, cube_size):
        print(pos.shape)
        assert pos.shape[0] == len(self.grid.shape)
        pos_start = pos - cube_size / 2
        pos_end = pos + cube_size / 2
        s = slice(pos_start, pos_end)
        return self.grid[s]


if __name__ == '__main__':
    f = Field(np.array([100, 100, 100]))
    f.check_collision(np.array([20, 20, 20]), 4)
