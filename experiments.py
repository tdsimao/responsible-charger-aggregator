from mdp import MDP
from Fleet import Fleet
from EV import EV
import numpy as np
import pandas as pd
from grid import Grid


def init_ev_fleet(num_charge_timesteps, grid_pos_list, deadline):
	return initialize_identical_EV_fleet(init_batt_level=0, batt_max=num_charge_timesteps, charge_rate=1, grid_pos_list= grid_pos_list,deadline=deadline)

def initialize_identical_EV_fleet(init_batt_level, batt_max, charge_rate, grid_pos_list, deadline):
	fleet = Fleet()
	for ev_pos in grid_pos_list:
		fleet.add_vehicle(EV(init_batt_level, batt_max, batt_max, charge_rate, ev_pos, deadline))
	return fleet



#  EXPERIMENT 1
LOW_PRICE = 30
HIGH_PRICE = 200
FIRST_PRICE = 70

def get_prices(timestep):
	if timestep < 3:
		return [FIRST_PRICE]
	if timestep < 7:
		return [LOW_PRICE, HIGH_PRICE]
	if timestep < 12:
		return [90]
	return [float('inf')]


def get_parameterized_price_transition_probability_func(p):
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

def time_dependent_price_transition_probability_func(p):
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


def run_experiment1(fleet, output_file):
	grid_file = 'grids/grid_1.txt'
	STEP_SIZE = 0.02
	results_time_independent = {}
	for p in np.arange(0.0, 1 + STEP_SIZE, STEP_SIZE):
		mdp = MDP(fleet, grid_file, 12,
				  get_prices_func=get_prices,
				  price_transition_probability_func=get_parameterized_price_transition_probability_func(p))
		policy, expected_val = mdp.value_iteration()
		results_time_independent[p] = expected_val[0][0]

	results_time_dependent = {}
	for p in np.arange(0.0, 1 + STEP_SIZE, STEP_SIZE):
		mdp = MDP(fleet, grid_file, 12,
				  get_prices_func=get_prices,
				  price_transition_probability_func=time_dependent_price_transition_probability_func(p))
		policy, expected_val = mdp.value_iteration()
		results_time_dependent[p] = expected_val[0][0]

	data_frame = pd.DataFrame.from_dict(
		{"time independent": results_time_independent, "time dependent": results_time_dependent})
	data_frame.to_csv(output_file)


def plot_experiment(data_file, output_file):
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


def run_experiment2(output_file):
	grid = Grid.create_tree_grid(high=3, branch_factor=2)
	# increase number of veehicles
	for i in range(1, grid.n_nodes):
		fleet = init_ev_fleet(4, range(1, i+1), 12)
		mdp = MDP(fleet, grid, 12,
				  get_prices_func=get_prices)
		policy, expected_val = mdp.value_iteration()
		print(i, expected_val[0][0])

if __name__ == "__main__":
	run_experiment1(fleet=init_ev_fleet(4, [1, 2], 12), output_file="data/experiment1_fleet1.csv")
	plot_experiment(data_file="data/experiment1_fleet1.csv", output_file='data/experiment1_fleet1.pdf')

	run_experiment1(fleet=init_ev_fleet(4, [1, 1], 12), output_file="data/experiment1_fleet2.csv")
	plot_experiment(data_file="data/experiment1_fleet2.csv", output_file='data/experiment1_fleet2.pdf')


	run_experiment2(output_file='data/experiment2.csv')
	# grid.save_to_file("grids/grid_teste.txt")
	# print(grid.n_nodes)
