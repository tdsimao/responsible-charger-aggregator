import numpy as np


class Grid(object):
    def __init__(self, n_nodes, n_lines, lines, reactances, line_bounds):
        self.n_nodes = n_nodes
        self.nodes = range(self.n_nodes)
        self.n_lines = n_lines
        self.lines = lines
        self.line_bounds = self._get_line_bounds(line_bounds)
        self.x = self._get_x(reactances)
        self.S = self._compute_ptdfs()

    def _get_x(self, reactances):
        return self._list_to_matrix(reactances)

    def _get_line_bounds(self, line_bounds):
        return self._list_to_matrix(line_bounds)

    def _list_to_matrix(self, alist):
        matrix = np.zeros([self.n_nodes, self.n_nodes])
        for (i, j), bound in zip(self.lines, alist):
            matrix[i][j] = bound
        return matrix

    @staticmethod
    def load_grid_from_file(file_name):
        with open(file_name, "r") as f:
            first_line = f.readline().split("=")
            assert first_line[0] == "numBus"
            n_nodes = int(first_line[1])

            second_line = f.readline().split("=")
            assert second_line[0] == "numLines"
            n_lines = second_line[1]

            comment_line = f.readline()
            assert comment_line[0] == "#"

            lines = []
            reactances = []
            line_bounds = []
            for line in f.readlines():
                i, j, _, reactance, line_bound = line.split()
                lines.append((int(i), int(j)))
                reactances.append(float(reactance))
                line_bounds.append(float(line_bound))
        return Grid(n_nodes=n_nodes,
                    n_lines=n_lines,
                    lines=lines,
                    reactances=reactances,
                    line_bounds=line_bounds)

    def save_to_file(self, file_name):
        with open(file_name, 'w') as f:
            f.write("numBus={}\n".format(self.n_nodes))
            f.write("numLines={}\n".format(self.n_lines))
            f.write("# LINE DATA (from bus, to bus, circuit id, reactance x in p.u., MVA rating)\n")
            for i, j in self.lines:
                f.write("{} {} 0 {} {}\n".format(i, j, self.x[i, j], self.line_bounds[i, j]))

    def save_to_dot_file(self, output_file):
        s = "graph{\n"
        for i, j in self.lines:
            bound = int(self.line_bounds[i, j])
            reactance = self.x[i, j]
            s += "\"n{}\" -- \"n{}\" [label = \"{}, {}\"];\n".format(i, j, bound, reactance)
        s += "}\n"
        with open(output_file, 'w') as f:
            f.write(s)

    def save_to_dot_file_with_fleet(self, fleet, output_file):
        vehicles_per_node = dict()
        for vehicle in fleet.vehicles:
            vehicles_per_node[vehicle.grid_position] = vehicles_per_node.get(vehicle.grid_position, 0) + 1

        s = "graph{\n"
        for i, j in self.lines:
            bound = int(self.line_bounds[i, j])
            reactance = self.x[i, j]
            label_i = "n{}: {}".format(i, vehicles_per_node.get(i, 0))
            label_j = "n{}: {}".format(j, vehicles_per_node.get(j, 0))
            s += "\"{}\" -- \"{}\" [label = \"{}, {}\"];\n".format(label_i, label_j, bound, reactance)
        s += "}\n"
        with open(output_file, 'w') as f:
            f.write(s)

    @staticmethod
    def create_tree_grid(high, branch_factor, reactance=.1, line_bound=100):
        assert high > 1
        current_node = 0
        last_nodes = [0]
        lines, reactances, line_bounds = [], [], []
        for i in range(1, high):
            new_nodes = []
            for node in last_nodes:
                for j in range(branch_factor):
                    current_node += 1
                    lines.append((node, current_node))
                    reactances.append(reactance)
                    line_bounds.append(line_bound)
                    new_nodes.append(current_node)
            last_nodes = new_nodes
        return Grid(n_nodes=current_node+1,
                    n_lines=len(lines),
                    lines=lines,
                    reactances=reactances,
                    line_bounds=line_bounds)

    def _compute_ptdfs(self):
        """
        Precomputing Power Transfer Distribution Factors
        """
        z = self._compute_z()
        s = np.zeros([self.n_nodes, self.n_nodes, self.n_nodes])
        for k in range(self.n_nodes):
            for L in range(self.n_nodes):
                for i in range(self.n_nodes):
                    if k == 0 and L != 0:
                        s[k, L, i] = -1 * z[L-1, i-1]
                    elif k != 0 and L == 0:
                        s[k, L, i] = z[k-1, i-1]
                    elif k != 0 and L != 0 and k != L:
                        s[k, L, i] = z[k-1, i-1] - z[L-1, i-1]
        return s

    def _compute_z(self):
        m = self._compute_m()
        return np.linalg.inv(m[1::, 1::])

    def _compute_m(self):
        m = np.zeros([self.n_nodes, self.n_nodes])
        for i, j in self.lines:
            x = self.x[i, j]
            m[i][j] += 1.0 / x
            m[j][i] += 1.0 / x
            m[i][i] += 1.0 / x
            m[j][j] += 1.0 / x
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if i != j:
                    m[i][j] = -1.0 * m[i][j]
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
            flow[k, l] = self.line_flow(k, l, loads)
        return flow

    def line_flow(self, k, l, loads):
        return sum(loads[i] * 1. / self.x[k, l] * self.S[k, l, i] for i in range(1, self.n_nodes))

    def feasible(self, loads):
        flow = self.compute_flow(loads)
        for i, j in self.lines:
            if abs(flow[i, j]) > self.line_bounds[i, j]:
                return False
        return True


def test_grid_feasibility(testing_grid, loads, expected_result):
    assert testing_grid.feasible(loads) == expected_result


if __name__ == "__main__":
    """
    running some tests with the example from [Walraven and Morales-España, 2015]
    """
    grid = Grid.load_grid_from_file('grids/grid_1.txt')
    grid.save_to_dot_file('grids/grid_1.dot')

    print(grid.compute_flow([2, 1, -3]))

    test_grid_feasibility(testing_grid=grid, loads=[2, -1, 3], expected_result=True)
    test_grid_feasibility(testing_grid=grid, loads=[-400, 200, 200], expected_result=True)
    test_grid_feasibility(testing_grid=grid, loads=[401, -201, -200], expected_result=False)
    test_grid_feasibility(testing_grid=grid, loads=[401, -200, -201], expected_result=False)
    test_grid_feasibility(testing_grid=grid, loads=[-200, 401, -201], expected_result=False)
