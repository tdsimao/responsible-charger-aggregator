from gurobipy import *


def solve_lp(model, show_results=True):
    if not show_results:
        model.Params.OutputFlag = 0
    model.optimize()

    if model.status == GRB.Status.INF_OR_UNBD:
        # Turn presolve off to determine whether model is infeasible
        # or unbounded
        model.setParam(GRB.Param.Presolve, 0)
        model.optimize()

    if model.status == GRB.Status.OPTIMAL:
        # model.write('model.sol')
        if not show_results:
            return model
        print('Optimal objective: %g' % model.objVal)
        for v in model.getVars():
            print("{:<25} {:<15}".format(v.varName, v.x))
        return
    elif model.status != GRB.Status.INFEASIBLE:
        print('Optimization was stopped with status %d' % model.status)
        return


def print_lp(m):
    m.update()
    m.write('model.lp')
    with open('model.lp') as f:
        print(f.read())
