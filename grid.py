import numpy as np
from util import *
from gurobipy import *


class Grid(object):
    def __init__(self, n_nodes, n_lines, lines, reactances):
        self.n_nodes = n_nodes
        self.n_lines = n_lines
        self.lines = lines
        self.x = self._get_x(reactances)
        self.S = self._compute_ptdfs()

    def _get_x(self, reactances):
        x = np.zeros([self.n_nodes, self.n_nodes])
        for (i, j), r in zip(self.lines, reactances):
            x[i][j] = r
            x[j][i] = r
        return x

    # @staticmethod
    # def load_grid_from_file(self):
    #     return Grid()


    def _compute_ptdfs(self):
        """
        Precomputing Power Transfer Distribution Factors
        """
        z = self._compute_z()
        s = np.zeros([self.n_nodes, self.n_nodes, self.n_nodes])
        for i in range(self.n_nodes):
            for l in range(self.n_nodes):
                for k in range(self.n_nodes):
                    if k == 0 and l != 0:
                        s[k, l, i] = -1 * z[l-1, i-1]
                    elif k != 0 and l == 0:
                        s[k, l, i] = z[k-1, i-1]
                    elif k != 0 and l != 0 and k != l:
                        s[k, l, i] = z[k-1, i-1] - z[l-1, i-1]
        return s

    def _compute_z(self):
        m = self._compute_m()
        return np.linalg.inv(m[1::, 1::])

    def _compute_m(self):
        m = np.zeros([self.n_nodes,self.n_nodes])
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if i != j:
                    m[i, j] = -1./self.x[i, j]
                else:
                    m[i, j] = sum(1./self.x[k, j] for k in range(self.n_nodes) if k != j)
        return m

    def compute_flow(self, loads):
        """
        use precomputed ptdfs to calculate the flow in the grid
        :param loads: vector of load in each node
        :return: flow of the grid given the loads in each node
        """
        assert len(loads) == self.n_nodes
        flow = np.zeros([self.n_nodes, self.n_nodes])
        for k, l in self.lines:
            flow[k, l] = sum(loads[i] * 1./self.x[k, l] * self.S[k, l, i] for i in range(self.n_nodes-1))
        return flow


if __name__ == "__main__":
    grid = Grid(n_nodes=3,
                n_lines=3,
                lines=[(0, 1), (0, 2), (1, 2), (1, 0), (2, 0), (2, 1)],
                reactances=[.1, .1, .1])
    grid_flow = grid.compute_flow([2, 1, -3])
    print(grid_flow)
