from gurobipy import *

from util import *


class MDP(object):
    def __init__(self, horizon):
        self.actions = ["left", "right"]
        self.states = range(3)
        self.p_s0 = [1, 0, 0]
        self.horizon = horizon
        self.time_steps = range(self.horizon)

        self._transition = {}
        self._transition["right"] = [[.1, 0.9, 0], [0, .1, .9], [0, 0, 1]]
        self._transition["left"] = [[1, 0, 0], [.9, .1, 0], [0, .9, .1]]

        self._reward = {}
        self._reward["left"] = [0, 0, 0]
        self._reward["right"] = [0, 0, 1]

    def transition(self, state, action, successor, time_step):
        assert action in self.actions
        return self._transition[action][state][successor]

    def reward(self, state, action, time_step):
        assert action in self.actions
        return self._reward[action][state]


class CMDP(MDP):
    def __init__(self, horizon):
        super(CMDP, self).__init__(horizon)

    def getLP(self):
        lp_model = Model("CMDP")

        x = list()
        for t in self.time_steps:
            x.append([])
            for s in self.states:
                varname = "x[" + str(t) + "][" + str(s) + "]"
                x[t].append(lp_model.addVars(self.actions, name=varname))

        lp_model.setObjective(quicksum(quicksum(quicksum(
                              x[t][s][a] * self.reward(s, a, t)
                              for t in self.time_steps)
                              for s in self.states)
                              for a in self.actions),
                              GRB.MAXIMIZE)

        # set probability mass at time step 0
        for s in self.states:
            prob_s = quicksum(x[0][s][a] for a in self.actions)
            lp_model.addConstr(prob_s == self.p_s0[s])

        # set probability mass for initial states at time step 0
        for suc in self.states:
            for t in self.time_steps[:-1]:
                prob_out = quicksum(x[t+1][suc][a] for a in self.actions)
                prob_in = quicksum(quicksum(
                            self.transition(s, a, suc, t) * x[t][s][a]
                            for s in self.states)
                            for a in self.actions)
                lp_model.addConstr(prob_in == prob_out)

        return lp_model

if __name__ == "__main__":
    m = CMDP(horizon=4)
    print(m.transition(0, "right", 1, 0))
    print(m.reward(0, "right", 0))
    print(m.states)
    print(m.actions)
    lp = m.getLP()
    print_lp(lp)

    solve_lp(lp)
