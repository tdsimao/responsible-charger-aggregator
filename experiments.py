from mdp import MDP
from Fleet import Fleet
from EV import EV
import numpy as np
import pandas as pd
from grid import Grid
import matplotlib.pyplot as plt


def init_ev_fleet(num_charge_timesteps, grid_pos_list, deadline):
	return initialize_identical_ev_fleet(init_batt_level=0,
										 batt_max=num_charge_timesteps,
										 charge_rate=1,
										 grid_pos_list= grid_pos_list,
										 deadline=deadline)


def initialize_identical_ev_fleet(init_batt_level, batt_max, charge_rate, grid_pos_list, deadline):
	fleet = Fleet()
	for ev_pos in grid_pos_list:
		fleet.add_vehicle(EV(init_batt_level, batt_max, batt_max, charge_rate, ev_pos, deadline))
	return fleet


def get_prices(timestep):
	if timestep < 4:
		return [FIRST_PRICE]
	if timestep < 8:
		return [HIGH_PRICE, LOW_PRICE]
	if timestep < 12:
		return [90]
	return [float('inf')]


def price_transition_probability_func(p):
	"""
	create a new function that return a price transition function
	if there is two future prices there is a probability p of going to the lower price
	and 1 - p of going to the higher price\

	this function ignores the history of the prices
	"""
	def price_transition_probability(from_price, to_price, timestep=0):
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
	def price_transition_probability(from_price, to_price, timestep=0):
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

def show_prices(get_prices_func, price_transition_probability_func, output_file):
	s = "digraph{\nrankdir=\"LR\";\n"
	for i in range(HORIZON):
		prices = get_prices_func(i)
		for p in prices:
			for fut_price in get_prices(i+1):
				prob = price_transition_probability_func(p, fut_price, i)
				if prob > 0:
					print("{:5} | {:5} | {:5} | {:5}".format(i, p, fut_price, prob))
					s += "\"{}.{}\" -> \"{}.{}\" [label = \"{}\", weight = \"{}\"];\n".format(i, p, i+1, fut_price, prob, prob)
	s += "}\n"
	with open(output_file, 'w') as f:
		f.write(s)



#
#  EXPERIMENT 1
#

LOW_PRICE = 30
HIGH_PRICE = 200
FIRST_PRICE = 70


def run_experiment1(fleet, output_file):
	grid_file = 'grids/grid_1.txt'
	STEP_SIZE = 0.02
	results_time_independent = {}
	for p in np.arange(0.0, 1 + STEP_SIZE, STEP_SIZE):
		mdp = MDP(fleet, grid_file, 12,
				  get_prices_func=get_prices,
				  price_transition_probability_func=price_transition_probability_func(p))
		policy, expected_val = mdp.value_iteration()
		results_time_independent[p] = expected_val[0][0]

	results_time_dependent = {}
	for p in np.arange(0.0, 1 + STEP_SIZE, STEP_SIZE):
		mdp = MDP(fleet, grid_file, 12,
				  get_prices_func=get_prices,
				  price_transition_probability_func=history_dependent_price_transition_probability_func(p))
		policy, expected_val = mdp.value_iteration()
		mdp.print_policy_expected_value(policy, expected_val)
		results_time_dependent[p] = expected_val[0][0]

	data_frame = pd.DataFrame.from_dict(
		{"time independent": results_time_independent, "time dependent": results_time_dependent})
	data_frame.to_csv(output_file)


def plot_experiment1(data_file, output_file):
	data_frame = pd.DataFrame.from_csv(data_file)

	ax = data_frame.plot()

	fig = ax.get_figure()
	ax.spines["top"].set_visible(False)
	ax.spines["right"].set_visible(False)
	ax.spines["bottom"].set_visible(False)
	ax.spines["left"].set_visible(False)
	ax.yaxis.grid(True)
	ax.xaxis.grid(False)

	fig.tight_layout()
	fig.savefig(output_file, bbox_inches='tight')
	print("plot saved at: " + output_file)


#
# Experiment 2
#
HORIZON = 12
MAX_NUMBER_OF_CARS = 8
TREE_HIGH = 2
TREE_BRANCHING_FACTOR = 2
VEHICLES_PER_LINE= 2 #number


def run_experiment2(grid, output_file):


	# increasing number of vehicles
	print("Number of nodes in the grid: {}".format(grid.n_nodes))
	expected_val_initial_state = {}
	profit_increase_rate = {1:0}
	for num_vehicles in range(1, MAX_NUMBER_OF_CARS+1):
		pos_vehicles = [i%(grid.n_nodes - 1) + 1 for i in range(num_vehicles)]
		print(pos_vehicles)
		fleet = init_ev_fleet(4, pos_vehicles, HORIZON)
		mdp = MDP(fleet, grid, HORIZON, get_prices_func=get_prices)
		policy, expected_val = mdp.value_iteration()
		print(num_vehicles, expected_val[0][0])
		expected_val_initial_state[num_vehicles] = expected_val[0][0]
		if num_vehicles > 1:
			profit_increase_rate[num_vehicles] = expected_val_initial_state[num_vehicles] /expected_val_initial_state[num_vehicles -1]

		data_frame = pd.DataFrame.from_dict(
			{"Expected value": expected_val_initial_state, "Profit increase rate": profit_increase_rate},
		)
		data_frame.to_csv(output_file)



def plot_experiment2(data_file, output_file):
	data_frame = pd.DataFrame.from_csv(data_file)
	# ax = data_frame.plot.bar()
	# fig = plt.figure()

	ax = data_frame.plot(secondary_y=['Profit increase rate'])
	# ax = data_frame.plot(secondary_y=True, style='g')

	# ax.set_ylabel('Profit increase rate')
	# ax.right_ax.set_ylabel('Profit increase rate')
	fig = ax.get_figure()
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
	args = parser.parse_args()

	if 1 in args.experiments:
		print("Experiment 1")
		run_experiment1(fleet=init_ev_fleet(5, [1, 2], 12), output_file="out/experiment1_fleet1.csv")
		run_experiment1(fleet=init_ev_fleet(5, [1, 1], 12), output_file="out/experiment1_fleet2.csv")

	if 1 in args.plots:
		plot_experiment1(data_file="out/experiment1_fleet1.csv", output_file='out/experiment1_fleet1.pdf')
		plot_experiment1(data_file="out/experiment1_fleet2.csv", output_file='out/experiment1_fleet2.pdf')

	if 2 in args.experiments:
		print("Experiment 2")
		grid = Grid.create_tree_grid(high=TREE_HIGH, branch_factor=TREE_BRANCHING_FACTOR,
									 line_bound=200 * VEHICLES_PER_LINE)
		run_experiment2(grid, output_file='out/experiment2.csv')

	if 2 in args.plots:
		plot_experiment2(data_file="out/experiment2.csv", output_file='out/experiment2.pdf')

	if args.render_prices:
		print("Experiment 3")
		show_prices(get_prices, price_transition_probability_func(.6), "out/price_transition_probability_func.dot")
		print()
		show_prices(get_prices, history_dependent_price_transition_probability_func(.6), "out/history_dependent_price_transition_probability_func.dot")

