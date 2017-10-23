from mdp import MDP
from Fleet import Fleet
from EV import EV

def init_ev_fleet(num_charge_timesteps, grid_pos_list, deadline):
	return initialize_identical_EV_fleet(0, num_charge_timesteps, 1, grid_pos_list, deadline)

def initialize_identical_EV_fleet(init_batt_level, batt_max, charge_rate, grid_pos_list, deadline):
	fleet = Fleet()
	for i in range(len(grid_pos_list)):
		fleet.add_vehicle(EV(init_batt_level, batt_max, batt_max, charge_rate, grid_pos_list[i], deadline))
	print(fleet)
	return fleet



# These functions can be adjusted and sent to the mdp class as parameters to be used
def get_prices(timestep):
	# TODO replace with actual prices
	if timestep < 3:
		return [70]
	if timestep < 7:
		return [30]
	if timestep < 12:
		return [90]
	return [float('inf')]

def price_transition_probability(from_price, to_price, timestep=0):
	#TODO replace with actual prob of going from price to price at timestep
	return 1.0 # Placeholder value


if __name__ == "__main__":
	fl = init_ev_fleet(6, [1,2], 12)

	grid_file = 'grids/grid_1.txt'

	mdp = MDP(fl, grid_file, 12, get_prices_func=get_prices, price_transition_probability_func=price_transition_probability)
	# mdp.print_transition_table()
	mdp.print_policy_expected_value()
	print(mdp.fleet)

	# test_state_to_list_to_state(mdp)
	# test_action_to_list(mdp)
	# test_state_plus_action(mdp)

	pass

