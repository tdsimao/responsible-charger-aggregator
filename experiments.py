from mdp import MDP, UncoordinatedMDP
from Fleet import Fleet
from EV import EV
from grid import Grid
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


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


def get_history_independent_price_transition_func(p):
    """
    create a new function that returns a price transition function
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


def get_history_dependent_price_transition_func(p):
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


def run_experiment1(fleet, grid, horizon, output_file):
    results_time_independent = experiment_low_price_prob(fleet, grid, horizon,
                                                         price_transition=get_history_independent_price_transition_func)

    results_time_dependent = experiment_low_price_prob(fleet, grid, horizon,
                                                       price_transition=get_history_dependent_price_transition_func)

    data_frame = pd.DataFrame.from_dict(
        {"History independent": results_time_independent,
         "History dependent": results_time_dependent})
    data_frame.to_csv(output_file)


def experiment_low_price_prob(fleet, grid, horizon, price_transition, step_size=0.01):
    restults = {}
    for p in np.arange(0.0, 1 + step_size, step_size):
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


def run_experiment2(line_bound, horizon, output_file):
    # increasing number of vehicles
    ev_s0 = {}
    profit_increase_rate = {1: 1}
    pre_processing_time = {}
    processing_time = {}
    average_reward = {}
    error_reward = {}

    initial_time = time.time()
    grid = Grid.create_tree_grid(high=TREE_HIGH, branch_factor=TREE_BRANCHING_FACTOR,
                                  line_bound=line_bound)
    grid_initialization_time = time.time() - initial_time

    for num_vehicles in range(1, MAX_NUMBER_OF_CARS + 1):
        pos_vehicles = [i % (grid.n_nodes - 1) + 1 for i in range(num_vehicles)]
        fleet = init_ev_fleet(4, pos_vehicles, horizon)
        grid.save_to_dot_file_with_fleet(fleet, "grids/grid_experiment2_fleet{}.dot".format(num_vehicles))

        mdp = MDP(fleet, grid, horizon, get_prices_func=get_prices)
        results = mdp.solve_get_stats()

        print(num_vehicles, results)

        pre_processing_time[num_vehicles] = results["Feasible actions computational time"] + grid_initialization_time
        processing_time[num_vehicles] = results["Optimization time"]
        ev_s0[num_vehicles] = results["Expected value initial state"]
        average_reward[num_vehicles] = results["average_reward"]
        error_reward[num_vehicles] = results["error"]
        if num_vehicles > 1:
            profit_increase_rate[num_vehicles] = (ev_s0[num_vehicles] - ev_s0[num_vehicles - 1]) / ev_s0[1]

        data_frame = pd.DataFrame.from_dict(
            {"Expected value": ev_s0,
             "Profit increase rate": profit_increase_rate,
             "Optimization": processing_time,
             "Average reward": average_reward,
             "error_reward": error_reward,
             "Preprocessing": pre_processing_time
             }
        )
        data_frame.to_csv(output_file)


def experiment2_plot_expected_value(data_file, output_file):
    data_frame = pd.DataFrame.from_csv(data_file)
    data_frame = data_frame[["Expected value", "Profit increase rate"]]

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
    data_frame = pd.DataFrame.from_csv(data_file)[["Optimization", "Preprocessing"]]
    ax = data_frame.plot(logy=True)

    ax.set_xlabel('Number of vehicles connected to the grid')
    ax.set_ylabel('Time (s)')

    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(output_file)

    print("plot saved at: " + output_file)

#
# Experiment 3
#


def run_experiment3(grid, fleet, horizon):
    # solve mdp and run a simulation
    # returns profile of the simulation

    mdp = MDP(fleet, grid, horizon, get_prices_func=deterministic_prices)
    profile_mdp_simulation(mdp, "out/experiment3_coordinated.csv")

    mdp = UncoordinatedMDP(fleet, grid, horizon, get_prices_func=deterministic_prices)
    profile_mdp_simulation(mdp, "out/experiment3_uncoordinated.csv")


def profile_mdp_simulation(mdp, output_file):
    policy, _ = mdp.value_iteration()
    results = mdp.run_simulation(initial_state=0, policy=policy)

    parsed_results = {k: results[k] for k in ["total_loads", "rewards", "accumulated_reward", "prices"]}
    for node in mdp.grid.nodes:
        parsed_results["load_node_{}".format(node)] = [results["loads"][i][node] for i in range(mdp.horizon)]
    for n1, n2 in mdp.grid.lines:
        flows = [abs(results["flows"][i][n1, n2]) for i in range(mdp.horizon)]
        parsed_results["flow_line_{}_{}".format(n1, n2)] = flows

    data_frame = pd.DataFrame.from_dict(parsed_results)
    data_frame.to_csv(output_file)


def experiment3_plot_flows(key_to_file, grid, output_file):
    fig, axes = plt.subplots(nrows=len(grid.lines)+1, ncols=1, sharex=True)

    axes[0].set_ylabel('Energy price ($ MW/h)')
    prices_df = pd.DataFrame.from_csv(next(iter(key_to_file.values())))["prices"]
    prices_df.plot(ax=axes[0])
    axes[0].set_title('Energy prices')

    for i, (n1, n2) in enumerate(grid.lines, start=1):
        flow_column = "flow_line_{}_{}".format(n1, n2)

        data = {}
        for model, file in key_to_file.items():
            df = pd.DataFrame.from_csv(file)
            data[model] = df[[flow_column]].T.iloc[0]
        new_df = pd.DataFrame.from_dict(data)
        new_df.plot.bar(ax=axes[i], legend=True)

        axes[i].set_title(flow_column.replace("_",  " ").capitalize())
        axes[i].set_ylabel('Power flow')

        axes[i].axhline(y=grid.line_bounds[n1, n2], c="blue", linewidth=0.75, linestyle='--', label="Line bound")
        axes[i].legend()
    axes[-1].set_xlabel('Time step')

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
    parser.add_argument('--line_capacity', type=int, default=420,
                        help='power capacity of each line')
    args = parser.parse_args()

    if 1 in args.experiments:
        print("Experiment 1")
        grid1 = Grid.load_grid_from_file("grids/grid_1.txt")
        fleet1 = init_ev_fleet(5, [1, 2], args.horizon)
        run_experiment1(fleet=fleet1, grid=grid1, horizon=args.horizon,
                        output_file="out/experiment1_fleet1.csv")
        grid1.save_to_dot_file_with_fleet(fleet1, "grids/grid_experiment1_fleet1.dot")

        fleet2 = init_ev_fleet(5, [1, 1], args.horizon)
        run_experiment1(fleet=fleet2, grid=grid1, horizon=args.horizon,
                        output_file="out/experiment1_fleet2.csv")
        grid1.save_to_dot_file_with_fleet(fleet2, "grids/grid_experiment1_fleet2.dot")

    if 1 in args.plots:
        plot_experiment1(data_file="out/experiment1_fleet1.csv", output_file='out/experiment1_fleet1.pdf')
        plot_experiment1(data_file="out/experiment1_fleet2.csv", output_file='out/experiment1_fleet2.pdf')

    if 2 in args.experiments:
        print("Experiment 2")

        run_experiment2(line_bound=args.line_capacity, horizon=args.horizon, output_file='out/experiment2.csv')

    if 2 in args.plots:
        experiment2_plot_expected_value(data_file="out/experiment2.csv", output_file='out/experiment2.pdf')
        experiment2_plot_processing_time(data_file="out/experiment2.csv",
                                         output_file='out/experiment2processing_time.pdf')

    if 3 in args.experiments:
        print("Experiment 3")
        grid3 = Grid.create_tree_grid(high=TREE_HIGH, branch_factor=1,
                                      line_bound=args.line_capacity)
        vehicles_positions = [i % (grid3.n_nodes - 1) + 1 for i in range(2)]
        fleet3 = init_ev_fleet(4, vehicles_positions, args.horizon)

        run_experiment3(grid=grid3, fleet=fleet3, horizon=args.horizon)
        grid3.save_to_dot_file_with_fleet(fleet3, "grids/grid_experiment3_fleet.dot")

    if 3 in args.plots:
        grid3 = Grid.create_tree_grid(high=TREE_HIGH, branch_factor=1,
                                      line_bound=args.line_capacity)
        model_to_file = {"Coordinated fleet": "out/experiment3_coordinated.csv",
                         "Uncoordinated fleet": "out/experiment3_uncoordinated.csv"}
        experiment3_plot_flows(key_to_file=model_to_file, grid=grid3, output_file='out/experiment3.pdf')

    if args.render_prices:
        print("Printing prices")
        show_prices(get_prices, get_history_independent_price_transition_func(.6), args.horizon,
                    "out/price_transition_probability_func.dot")
        print()
        show_prices(get_prices, get_history_dependent_price_transition_func(.6), args.horizon,
                    "out/history_dependent_price_transition_probability_func.dot")
        print()
        show_prices(deterministic_prices, price_transition_uniform, args.horizon,
                    "out/prices_deterministic.dot")
