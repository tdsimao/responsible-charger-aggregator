import numpy as np
from util import *
from gurobipy import *


class Grid(object):
    def __init__(self):
        self.n_nodes = 3
        self.n_lines = 3
        self.x = self.initialize_x()
        self.lines = self.init_lines()
        M = self.compute_m()
        self.S = self.compute_s()
        self.Z = self.compute_z()

    def init_lines(self):
        return [(0, 1), (0, 2), (1, 2), (1, 0), (2, 0), (2, 1)]

    def initialize_x(self):
        x = np.zeros([self.n_nodes, self.n_nodes])
        x[0][1] = .1
        x[0][2] = .1
        x[1][2] = .1
        x[1][0] = x[0][1]
        x[2][0] = x[0][2]
        x[2][1] = x[1][2]
        return x

    def compute_s(self):
        Z = self.compute_z()
        S = np.zeros([self.n_nodes, self.n_nodes, self.n_nodes])
        for i in range(self.n_nodes):
            for l in range(self.n_nodes):
                for k in range(self.n_nodes):
                    if k == 0 and l != 0:
                        S[k, l, i] = -1 * Z[l-1, i-1]
                    elif k != 0 and l == 0:
                        S[k, l, i] = Z[k-1, i-1]
                    elif k != 0 and l != 0 and k != l:
                        S[k, l, i] = Z[k-1, i-1] - Z[l-1, i-1]
        return S

    def compute_z(self):
        M = self.compute_m()
        return np.linalg.inv(M[1::, 1::])

    def compute_m(self):
        M = np.zeros([self.n_nodes,self.n_nodes])
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if i != j:
                    M[i, j] = -1./self.x[i, j]
                else:
                    M[i, j] = sum(1./self.x[k, j] for k in range(self.n_nodes) if k != j)
        return M

    def compute_flow(self, loads):
        """

        :param loads: vector of load in each node
        :return:
        """
        flow = np.zeros([self.n_nodes, self.n_nodes])
        for k, l in self.lines:
            flow[k, l] = sum(loads[i] * 1./self.x[k, l] * self.S[k, l, i] for i in range(self.n_nodes-1))
        return flow


if __name__ == "__main__":
    grid = Grid()
    grid_flow = grid.compute_flow([2, 1, -3])
    print(grid_flow)
