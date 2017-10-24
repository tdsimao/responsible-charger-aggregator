# Vincent Koeten
# Adapted HW1 value iteration

import numpy as np
from pprint import pprint
from math import pow, ceil
from functools import lru_cache
from EV import EV
from Fleet import Fleet
from grid import Grid

MAXIMUM_PRICE = 100
TRANSITION_TABLE_PRINT_FLOAT_FLAG = False

class MDP:

	def __init__(self, fleet, grid, horizon=12, get_prices_func=None, price_transition_probability_func=None):
		self.fleet = fleet

		if isinstance(grid, str):
			self.grid = Grid.load_grid_from_file(grid)
		else:
			self.grid = grid

		self.horizon = horizon
		# If pricing info is kept in state then multiply by nPrices
		self.num_states = 1
		for ev in self.fleet.vehicles:
			self.num_states *= ev.num_charge_steps
		self.num_charge_rates = 2 # Binary charging
		self.num_actions = int(pow(self.num_charge_rates, self.fleet.size()))
		self.feasible_actions = None
		self.feasible_actions = self.grid_feasible_actions()
		self.get_prices_func = get_prices_func
		self.price_transition_probability_func = price_transition_probability_func

	# Solve MDP and return optimal action
	# @param discountFactor the discount factor
	# @return optimal action to select in state s_0
	def value_iteration(self):
		"""
		:return: greedy policy for each time step and expected value of each state at timestep 0
		"""
		qn = np.zeros((len(self.get_prices(self.horizon)), self.num_states, self.num_actions))
		policy = []
		# The value iteration algorithm
		for t in reversed(range(self.horizon)):
			prices = self.get_prices(t)
			qnp1 = np.zeros((len(prices), self.num_states, self.num_actions))
			for price_ind, price in enumerate(prices):
				for s in self.get_states():
					for a in self.feasible_actions_in_state(s):
						expected_future_reward = 0
						for future_price_ind, future_price in enumerate(self.get_prices(t + 1)):
							expected_future_reward += self.future_expected_reward(qn[future_price_ind], s, a) * \
														self.price_transition_probability(price, future_price, timestep=t)
						qnp1[price_ind][s][a] = self.get_reward(s, a, price) + expected_future_reward
			qn = qnp1
			new_policy = []
			for price_ind, _ in enumerate(prices):
				new_policy.append([self.greedy_policy(qn[price_ind][s], s) for s in self.get_states()])
			policy.append(new_policy)

		expected_values = []
		for s in self.get_states():
			expected_values.append([])
			for price_ind, _ in enumerate(self.get_prices(0)):
				expected_values[s].append(max(self.future_expected_reward(qn[price_ind], s, a) for a in range(self.num_actions)))

		return policy[::-1], expected_values

	def greedy_policy(self, q, s):
		"""
		return list of greedy feasible actions
		"""
		result = []
		if q is None:
			return result
		max_val = q[0]
		for action in self.feasible_actions_in_state(s):
			q_value = q[action]
			if q_value == max_val:
				result.append(action)
			elif q_value > max_val:
				result = [action]
				max_val = q_value
		return result

	def future_expected_reward(self, qn, s, a):
		result = 0
		for prob, sp in self.future_states_probabilities(s, a):
			result += prob * max(qn[sp])
		return result

	def future_states_probabilities(self, from_state, action):
		return [(1,self.charge_from_state(from_state, action))]

	def grid_feasible_actions(self):
		if self.feasible_actions is None:
			self.feasible_actions = []
			for action in range(self.num_actions):
				if self.grid.feasible(self.get_load(action)):
					self.feasible_actions.append(action)
		return self.feasible_actions

	def get_states(self):
		return range(self.num_states)

	@lru_cache(maxsize=10000) #TODO set maxsize according to the number of states
	def feasible_actions_in_state(self, s):
		assert 0 <= s < self.num_states
		vehicles_charge = np.array(self.charge_state_to_list(s))
		max_charge = np.array(list(ev.battery_max for ev in self.fleet.vehicles))
		vehicles_charged = vehicles_charge == max_charge

		result = []
		for action in self.grid_feasible_actions():
			charging_vehicles = np.array(self.charge_action_to_list(action)).astype(bool)
			if not any(charging_vehicles & vehicles_charged):
				result.append(action)
		return result

	def get_prices(self, timestep):
		if self.get_prices_func is not None:
			return self.get_prices_func(timestep)
		else:
			# TODO placeholder function
			return [float('inf')]

	def transition_probability(self, action, from_state, to_state):
		#TODO
		if to_state == self.charge_from_state(from_state, action):
			return 1
		else:
			return 0

	def price_transition_probability(self, from_price, to_price, timestep=0):
		if self.price_transition_probability_func is not None:
			return self.price_transition_probability_func(from_price, to_price, timestep)
		else:
			#TODO placeholder function
			return 1.0/(len(self.get_prices(timestep)))

	def get_reward(self, state, action, price):
		# TODO
		# TODO avoid charging/rewarding charging full vehicles
		#Compute number of charging vehicles 
		charge_list = self.charge_action_to_list(action)
		num_evs_charging = sum(charge_list)
		#multiply by price set
		return (100 - price) * num_evs_charging

	def charge_state_to_list(self, charge_state):
		charge_list = []
		base_multiplier = 1
		for ev in self.fleet.vehicles:
			base_multiplier *= ev.num_charge_steps
		for i in range(self.fleet.size()):
			ev = self.fleet.vehicles[i]
			base_multiplier /= ev.num_charge_steps
			charge_list += [int(charge_state / base_multiplier)]
			charge_state %= base_multiplier
		return charge_list

	def charge_action_to_list(self, charge_action):
		# NOTE: Currently works for binary charging
		# TODO: Extend for additonal and possibly variable charge rates?
		action_list = []
		for ev in range(self.fleet.size()):
			# action_list = action_list + [(charge_action >> ev) & 1]
			action_list = [(charge_action >> ev) & 1] + action_list
		return action_list

	def charge_list_to_state(self, charge_list):
		if(len(charge_list) != self.fleet.size()):
			return None
		charge_state = 0
		base_multiplier = 1
		for i in range(self.fleet.size()-1, -1, -1):
			charge_state += base_multiplier * charge_list[i]
			base_multiplier *= self.fleet.vehicles[i].num_charge_steps
		return charge_state

	def charge_list_apply_action_list(self, charge_list, action_list):
		if(len(charge_list) != len(action_list)):
			# TODO throw error?
			return None
		for i in range(len(charge_list)):
			ev = self.fleet.vehicles[i]
			new_charge = charge_list[i] + action_list[i]
			if(new_charge <= ev.battery_max): # ensure that ev is not charging beyond capacity
				charge_list[i] = new_charge
			else:
				charge_list[i] = ev.battery_max
		return charge_list

	def charge_from_state(self, from_charge, charge_action):
		from_charge_list = self.charge_state_to_list(from_charge)
		charge_action_list = self.charge_action_to_list(charge_action)
		to_charge_list = self.charge_list_apply_action_list(from_charge_list, charge_action_list)
		to_charge = self.charge_list_to_state(to_charge_list)
		return to_charge

	def get_load(self, action):
		load = np.zeros(self.grid.n_nodes)
		for ev_ind, charging_rate in enumerate(self.charge_action_to_list(action)):
			if charging_rate > 0:
				ev = self.fleet.vehicles[ev_ind]
				power_consumption = 200.  # power consumption should be a parameter of the EV
				# multiply by charging in case of multiple chargin states
				load[ev.grid_position] += ev.charge_rate * power_consumption * charging_rate
		total_load = sum(load)
		load[0] = -total_load
		return load

	def print_transition_table(self):
		for action in range(self.num_actions):
			print("Action %2d" % action)
			for x in self.get_states():
				print("[", end='')
				for y in self.get_states():
					endstr = ""
					if(y < (self.num_states - 1)):
						endstr = ", "
					val = self.transition_probability(action, x, y)
					if TRANSITION_TABLE_PRINT_FLOAT_FLAG:
						formatstr = "%-4d"
						if val != 0:
							formatstr = "%1.2f" #%1.2f
						print(formatstr % val, end=endstr)
					else:
						if val == 0:
							print(" ", end = endstr)
						else:
							print("%d" % val, end=endstr)
				print("]")

	def print_policy_expected_value(self, policy, expected_value):
		for i in range(self.horizon):
			print("time step: | price | EVs Charging State |  best actions  ")
			for p_ind, price in enumerate(self.get_prices(i)):
				for s in range(self.num_states):
					actions = [self.charge_action_to_list(a) for a in policy[i][p_ind][s]]
					print("{:11}| {:5} | {:19}| {}".format(i, price, str(self.charge_state_to_list(s)), actions))
			print("")

		print("price | EVs Charging State | expected value time step 0")
		for p_ind, price in enumerate(self.get_prices(0)):
			for s in range(self.num_states):
				print("{:5} | {:19}| {}".format(price, str(self.charge_state_to_list(s)), expected_value[s][p_ind]))

################################################################################



def test_state_to_list_to_state(mdp):
	for state in range(mdp.num_states):
		print("state: %2d -> " % state, end='')
		csl = mdp.charge_state_to_list(state)
		print(csl, end=" -> ")
		print(mdp.charge_list_to_state(csl))

def test_action_to_list(mdp):
	for action in range(mdp.num_actions):
		print("action: %d -> " % action, end='')
		print(mdp.charge_action_to_list(action))

def test_state_plus_action(mdp):
	for state in range(mdp.num_states):
		for action in range(mdp.num_actions):
			print("state + action: %2d+%d -> " % (state, action), end='')
			csl = mdp.charge_list_apply_action_list(mdp.charge_state_to_list(state), mdp.charge_action_to_list(action))
			print(csl, end=" -> ")
			print(mdp.charge_list_to_state(csl))

#TODO extend following tests to work with current classes
def test_get_load():
	global evsList, nEVs, nChargeStates, nStates, nChargeRates, nActions, grid

	evsList =  [EV(0, 3, 3, 1, gridPos=2, deadline=23)]
	evsList += [EV(0, 4, 4, 1, gridPos=2, deadline=23)]

	nEVs = len(evsList)
	nChargeStates = 1
	for ev in evsList:
		nChargeStates *= ev.nChargeSteps
	nStates = nChargeStates # For now unless pricing info is kept in state then multiply by nPrices
	nChargeRates = 2
	nActions = int(pow(nChargeRates, nEVs))
	print("nEVs: %d, nPrices: %d, nChargeStates: %d, nStates: %d, nActions: %d" % (nEVs, nPrices, nChargeStates, nStates, nActions))

	grid = Grid.load_grid_from_file('grids/grid_1.txt')
	for action in range(nActions):
		print("action: ", action, chargeActionToList(action))
		load = get_load(evsList, action=action, grid=grid)
		print("load: ", load)
		print("flow", grid.compute_flow(load))
		if grid.feasible(load):
			print("feasible")
		else:
			print("not feasible")
		print("")

	mdp.print_policy_expected_value()

	solve(1)

def test_with_unfeasible_loads():
	evsList =  [EV(0, 3, 3, 1, gridPos=2, deadline=23)]
	evsList += [EV(0, 4, 4, 1, gridPos=2, deadline=23)]

	grid = Grid.load_grid_from_file('grids/grid_1.txt')
	for action in range(nActions):
		print("action: ", action, chargeActionToList(action))
		load = get_load(evsList, action=action, grid=grid)
		print("load: ", load)
		print("flow", grid.compute_flow(load))
		if grid.feasible(load):
			print("feasible")
		else:
			print("not feasible")
		print("")


if __name__ == "__main__":
	fl = Fleet()
	fl.add_vehicle(EV(0, 3, 3, 1, grid_pos=1, deadline=23))
	fl.add_vehicle(EV(0, 4, 4, 1, grid_pos=2, deadline=23))
	# fl.add_vehicle(EV(0, 4, 4, 1, 2, 23))

	grid_file = 'grids/grid_1.txt'

	mdp = MDP(fl, grid_file, 12)
	# mdp.print_transition_table()
	mdp.print_policy_expected_value()
	print(mdp.fleet)

	# test_state_to_list_to_state(mdp)
	# test_action_to_list(mdp)
	# test_state_plus_action(mdp)

	pass
