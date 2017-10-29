from mdp import MDP, UncoordinatedMDP
from Fleet import Fleet
from EV import EV
from grid import Grid
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def init_ev_fleet(num_charge_timesteps, grid_pos_list, deadline):
    return initialize_identical_ev_fleet(init_batt_level=0,
                                         batt_max=num_charge_timesteps,
                                         charge_rate=1,
                                         grid_pos_list=grid_pos_list,
                                         deadline=deadline)


def initialize_identical_ev_fleet(init_batt_level, batt_max, charge_rate, grid_pos_list, deadline):
    fleet = Fleet()
    for ev_pos in grid_pos_list:
        fleet.add_vehicle(EV(init_batt_level, batt_max, batt_max, charge_rate, ev_pos, deadline))
    return fleet


LOW_PRICE = 30
HIGH_PRICE = 200
FIRST_PRICE = 70
LAST_PRICE = 90


def get_prices(timestep):
    if timestep < 4:
        return [FIRST_PRICE]
    if timestep < 8:
        return [HIGH_PRICE, LOW_PRICE]
    if timestep < 12:
        return [LAST_PRICE]
    return [float('inf')]


def deterministic_prices(timestep):
    if timestep < 4:
        return [FIRST_PRICE]
    if timestep < 8:
        return [LOW_PRICE]
    if timestep < 12:
        return [LAST_PRICE]
    return [float('inf')]


def price_transition_uniform(from_price, to_price, timestep=0, get_price_func=get_prices):
    return 1./len(get_price_func(timestep + 1))


def price_transition_probability_func(p):
    """
    create a new function that return a price transition function
    if there is two future prices there is a probability p of going to the lower price
    and 1 - p of going to the higher price\

    this function ignores the history of the prices
    """

    def price_transition_probability(from_price, to_price, timestep=0, get_price_func=None):
        if len(get_prices(timestep + 1)) == 1:
            return 1
        if to_price == LOW_PRICE:
            return p
        if to_price == HIGH_PRICE:
            return 1 - p
        raise Exception("unknown future prices")

    return price_transition_probability


def history_dependent_price_transition_probability_func(p):
    """
    create a new function that return a price transition function
    if there is two future prices there is a probability p of going to the lower price
    and 1 - p of going to the higher price
    """

    def price_transition_probability(from_price, to_price, timestep=0, get_price_func=None):
        if len(get_prices(timestep + 1)) == 1:
            return 1
        if from_price == FIRST_PRICE:
            if to_price == LOW_PRICE:
                return p
            if to_price == HIGH_PRICE:
                return 1 - p
        else:
            if to_price == from_price:
                return 1
            else:
                return 0

    return price_transition_probability


def show_prices(get_prices_func, price_transition_func, horizon, output_file):
    s = "digraph{\nrankdir=\"LR\";\n"
    for i in range(horizon - 1):
        prices = get_prices_func(i)
        for p in prices:
            for fut_price in get_prices_func(i + 1):
                prob = price_transition_func(p, fut_price, i, get_prices_func)
                if prob > 0:
                    print("{:5} | {:5} | {:5} | {:5}".format(i, p, fut_price, prob))
                    s += "\"{}.{}\" -> \"{}.{}\" [label = \"{}\", weight = \"{}\"];\n".format(i, p, i + 1, fut_price,
                                                                                              prob, prob)
    s += "}\n"
    with open(output_file, 'w') as f:
        f.write(s)


#
#  EXPERIMENT 1
#


def run_experiment1(fleet, horizon, output_file):
    grid_file = 'grids/grid_1.txt'
    results_time_independent = exp_increasing_low_price_prob(fleet, grid_file, horizon,
                                                             price_transition=price_transition_probability_func)

    results_time_dependent = exp_increasing_low_price_prob(fleet, grid_file, horizon,
                                                           price_transition=history_dependent_price_transition_probability_func)

    data_frame = pd.DataFrame.from_dict(
        {"History independent": results_time_independent,
         "History dependent": results_time_dependent})
    data_frame.to_csv(output_file)


def exp_increasing_low_price_prob(fleet, grid, horizon, price_transition):
    restults = {}
    STEP_SIZE = 0.01
    for p in np.arange(0.0, 1 + STEP_SIZE, STEP_SIZE):
        mdp = MDP(fleet, grid, horizon,
                  get_prices_func=get_prices,
                  price_transition_func=price_transition(p))
        policy, expected_val = mdp.value_iteration()
        restults[p] = expected_val[0][0]
    return restults


def plot_experiment1(data_file, output_file):
    data_frame = pd.DataFrame.from_csv(data_file)

    ax = data_frame.plot(title="Expected value according to probability of low energy prices")

    fig = ax.get_figure()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)

    ax.set_ylabel('Expected value initial state')
    ax.set_xlabel('Probability of low energy prices')

    fig.tight_layout()
    fig.savefig(output_file, bbox_inches='tight')
    print("plot saved at: " + output_file)


#
# Experiment 2
#


MAX_NUMBER_OF_CARS = 6
TREE_HIGH = 2
TREE_BRANCHING_FACTOR = 2


def run_experiment2(grid, horizon, output_file):
    # increasing number of vehicles
    print("Number of nodes in the grid: {}".format(grid.n_nodes))
    ev_s0 = {}
    profit_increase_rate = {1: 1}
    processing_time = {}
    average_reward = {}
    error_reward = {}

    for num_vehicles in range(1, MAX_NUMBER_OF_CARS + 1):
        pos_vehicles = [i % (grid.n_nodes - 1) + 1 for i in range(num_vehicles)]
        fleet = init_ev_fleet(4, pos_vehicles, horizon)
        mdp = MDP(fleet, grid, horizon, get_prices_func=get_prices)
        results = mdp.solve_get_stats()

        print(num_vehicles, results)

        processing_time[num_vehicles] = results["Optimization time"]
        ev_s0[num_vehicles] = results["Expected value initial state"]
        average_reward[num_vehicles] = results["average_reward"]
        error_reward[num_vehicles] = results["error"]
        if num_vehicles > 1:
            profit_increase_rate[num_vehicles] = (ev_s0[num_vehicles] - ev_s0[num_vehicles - 1]) / ev_s0[1]

        data_frame = pd.DataFrame.from_dict(
            {"Expected value": ev_s0,
             "Profit increase rate": profit_increase_rate,
             "Optimization time": processing_time,
             "Average reward": average_reward,
             "error_reward": error_reward,
             }
        )
        data_frame.to_csv(output_file)


def experiment2_plot_expected_value(data_file, output_file):
    data_frame = pd.DataFrame.from_csv(data_file)
    data_frame = data_frame[["Expected value", "Average reward", 'Profit increase rate']]


    ax = data_frame.plot(secondary_y=['Profit increase rate'],
                         title="Diminishing profits for adding new vehicles")

    ax.set_xlabel('Number of vehicles connected to the grid')
    ax.set_ylabel('Expected value initial state')
    ax.right_ax.set_ylabel('Profit increase rate')

    ax.get_legend().set_bbox_to_anchor((0.5, .8))
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(output_file)

    print("plot saved at: " + output_file)


def experiment2_plot_processing_time(data_file, output_file):
    data_frame = pd.DataFrame.from_csv(data_file)[["Optimization time"]]
    ax = data_frame.plot(logy=True)

    ax.set_xlabel('Number of vehicles connected to the grid')
    ax.set_ylabel('Processing time')

    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(output_file)

    print("plot saved at: " + output_file)

#
# Experimet 3
#


def run_experiment3(grid, horizon, num_vehicles):
    # solve mdp and run a simulation
    # returns profile of the simulation

    pos_vehicles = [i % (grid.n_nodes - 1) + 1 for i in range(num_vehicles)]
    fleet = init_ev_fleet(4, pos_vehicles, horizon)

    mdp = MDP(fleet, grid, horizon, get_prices_func=deterministic_prices)
    profile_mdp_simulation(mdp, "out/experiment3_coordinated.csv")

    mdp = UncoordinatedMDP(fleet, grid, horizon, get_prices_func=deterministic_prices)
    profile_mdp_simulation(mdp, "out/experiment3_uncoordinated.csv")



def profile_mdp_simulation(mdp, output_file):
    policy, _ = mdp.value_iteration()
    results = mdp.run_simulation(initial_state=0, policy=policy)

    parsed_results = {k : results[k] for k in ["total_loads", "rewards", "accumulated_reward"]}
    for node in grid.nodes:
        parsed_results["load_node_{}".format(node)] = [results["loads"][i][node] for i in range(mdp.horizon)]
    for n1, n2 in grid.lines:
        flows = [abs(results["flows"][i][n1, n2]) for i in range(mdp.horizon)]
        parsed_results["flow_line_{}_{}".format(n1, n2)] = flows

    data_frame = pd.DataFrame.from_dict(parsed_results)
    data_frame.to_csv(output_file)


def experiment3_plot_flows(key_to_file, grid, output_file):
    fig, axes = plt.subplots(nrows=len(grid.lines), ncols=1)

    for i, (n1, n2) in enumerate(grid.lines):
        flow_column = "flow_line_{}_{}".format(n1, n2)

        data = {}
        for model, file in key_to_file.items():
            df = pd.DataFrame.from_csv(file)
            data[model] = df[[flow_column]].T.iloc[0]
        new_df = pd.DataFrame.from_dict(data)
        new_df.plot.bar(ax=axes[i], legend=True)

        axes[i].set_title(flow_column.replace("_",  " ").capitalize())
        axes[i].set_ylabel('Load')

    axes[i].set_xlabel('Time step')

    fig = axes[0].get_figure()
    fig.tight_layout()
    fig.savefig(output_file)

    print("plot saved at: " + output_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='Run the experiments.')
    parser.add_argument('--experiments', metavar='EXP_ID', type=int, nargs='*', default=[],
                        help='list of experiments to be executed')
    parser.add_argument('--plots', metavar='EXP_ID', type=int, nargs='*', default=[],
                        help='list of experiments to be plotted')
    parser.add_argument('--render_prices', action='store_true',
                        help='render price transition')
    parser.add_argument('--horizon', type=int, default=12,
                        help='horizon for the test')
    parser.add_argument('--vehicles_per_line_capacity', type=int, default=2,
                        help='number of vehicles that each line support')
    args = parser.parse_args()

    if 1 in args.experiments:
        print("Experiment 1")
        run_experiment1(fleet=init_ev_fleet(5, [1, 2], args.horizon), horizon=args.horizon,
                        output_file="out/experiment1_fleet1.csv")
        run_experiment1(fleet=init_ev_fleet(5, [1, 1], args.horizon), horizon=args.horizon,
                        output_file="out/experiment1_fleet2.csv")

    if 1 in args.plots:
        plot_experiment1(data_file="out/experiment1_fleet1.csv", output_file='out/experiment1_fleet1.pdf')
        plot_experiment1(data_file="out/experiment1_fleet2.csv", output_file='out/experiment1_fleet2.pdf')

    if 2 in args.experiments:
        print("Experiment 2")
        grid = Grid.create_tree_grid(high=TREE_HIGH, branch_factor=TREE_BRANCHING_FACTOR,
                                     line_bound=200 * args.vehicles_per_line_capacity)
        run_experiment2(grid, horizon=args.horizon, output_file='out/experiment2.csv')

    if 2 in args.plots:
        experiment2_plot_expected_value(data_file="out/experiment2.csv", output_file='out/experiment2.pdf')
        experiment2_plot_processing_time(data_file="out/experiment2.csv",
                                         output_file='out/experiment2processing_time.pdf')

    if 3 in args.experiments:
        print("Experiment 3")
        grid = Grid.create_tree_grid(high=TREE_HIGH, branch_factor=TREE_BRANCHING_FACTOR,
                                     line_bound=200 * args.vehicles_per_line_capacity)
        run_experiment3(grid=grid, horizon=args.horizon, num_vehicles=4)

    if 3 in args.plots:
        grid = Grid.create_tree_grid(high=TREE_HIGH, branch_factor=TREE_BRANCHING_FACTOR,
                                     line_bound=200 * args.vehicles_per_line_capacity)
        key_to_file = {"Coordinated fleet": "out/experiment3_coordinated.csv",
                       "Uncoordinated fleet": "out/experiment3_uncoordinated.csv"}
        experiment3_plot_flows(key_to_file=key_to_file, grid=grid, output_file='out/experiment3.pdf')

    if args.render_prices:
        print("Printing prices")
        show_prices(get_prices, price_transition_probability_func(.6), args.horizon,
                    "out/price_transition_probability_func.dot")
        print()
        show_prices(get_prices, history_dependent_price_transition_probability_func(.6), args.horizon,
                    "out/history_dependent_price_transition_probability_func.dot")
        print()
        show_prices(deterministic_prices, price_transition_uniform, args.horizon,
                    "out/prices_deterministic.dot")
