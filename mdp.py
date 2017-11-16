import random
import time

import numpy as np
from math import pow, ceil
from functools import lru_cache

from EV import EV
from Fleet import Fleet
from grid import Grid
from util import choose


TRANSITION_TABLE_PRINT_FLOAT_FLAG = False


class MDP:
	def __init__(self, fleet, grid, horizon=12, get_prices_func=None, price_transition_func=None, max_price=100):
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
		self.num_charge_rates = 2  # Binary charging
		self.num_actions = int(pow(self.num_charge_rates, self.fleet.size()))
		self.feasible_actions = None
		self.get_prices_func = get_prices_func
		self.price_transition_func = price_transition_func
		self.max_price = max_price

	def value_iteration(self):
		"""
		:return: greedy policy for each time step and expected value of each state and price at timestep 0
		"""
		qn = np.zeros((len(self.get_prices(self.horizon)), self.num_states, self.num_actions))
		greedy_policies = []

		for t in reversed(range(self.horizon)):
			prices = self.get_prices(t)
			qnp1 = np.zeros((len(prices), self.num_states, self.num_actions))
			for price_ind, price in enumerate(prices):
				for s in self.get_states():
					for a in self.feasible_actions_in_state(s):
						immediate_reward = self.get_reward(s, a, price)
						expected_future_reward = 0
						for future_price_ind, future_price in enumerate(self.get_prices(t + 1)):
							expected_future_reward += self.future_expected_reward(qn[future_price_ind], s, a) * \
														self.price_transition_probability(price, future_price, timestep=t)
						qnp1[price_ind][s][a] = immediate_reward + expected_future_reward
			qn = qnp1
			new_policy = self.greedy_police_prices(prices, qn)
			greedy_policies.append(new_policy)

		expected_values = self.get_expected_value(qn, 0)

		return greedy_policies[::-1], expected_values

	def get_expected_value(self, qn, timestep):
		expected_values = []
		for s in self.get_states():
			expected_values.append([])
			for price_ind, _ in enumerate(self.get_prices(timestep)):
				expected_values[s].append(
					max(self.future_expected_reward(qn[price_ind], s, a) for a in self.get_actions()))
		return expected_values

	def greedy_police_prices(self, prices, qn):
		new_policy = []
		for price_ind, _ in enumerate(prices):
			new_policy.append([self.greedy_policy(qn[price_ind][s], s) for s in self.get_states()])
		return new_policy

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
		for sp, prob in self.future_states_probabilities(s, a):
			result += prob * max(qn[sp])
		return result

	def future_states_probabilities(self, from_state, action):
		return [(self.charge_from_state(from_state, action), 1)]

	def grid_feasible_actions(self):
		if self.feasible_actions is None:
			self.feasible_actions = []
			for action in self.get_actions():
				if self.grid.feasible(self.get_load(action)):
					self.feasible_actions.append(action)
		return self.feasible_actions

	def get_states(self):
		return range(self.num_states)

	def get_actions(self):
		return range(self.num_actions)

	@lru_cache(maxsize=10000)  # TODO set maxsize according to the number of states
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
			return [30, 50]

	def transition_probability(self, action, from_state, to_state):
		if to_state == self.charge_from_state(from_state, action):
			return 1
		else:
			return 0

	def price_transition_probability(self, from_price, to_price, timestep=0):
		if self.price_transition_func is not None:
			return self.price_transition_func(from_price, to_price, timestep)
		else:
			return 1.0/(len(self.get_prices(timestep+1)))

	def future_prices_and_probabilities(self, from_price, timestep=0):
		result = []
		for ind, price in enumerate(self.get_prices(timestep+1)):
			prob = self.price_transition_probability(from_price, price, timestep)
			result.append((price, prob))
		return result

	def get_reward(self, state, action, price):
		charge_list = self.charge_action_to_list(action)
		num_evs_charging = sum(charge_list)
		if num_evs_charging == 0:
			return 0
		return (self.max_price - price) * num_evs_charging

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
		assert len(charge_list) == self.fleet.size()

		charge_state = 0
		base_multiplier = 1
		for i in range(self.fleet.size()-1, -1, -1):
			charge_state += base_multiplier * charge_list[i]
			base_multiplier *= self.fleet.vehicles[i].num_charge_steps
		return charge_state

	def charge_list_apply_action_list(self, charge_list, action_list):
		assert len(charge_list) == len(action_list)

		for i in range(len(charge_list)):
			ev = self.fleet.vehicles[i]
			new_charge = charge_list[i] + action_list[i]
			if new_charge <= ev.battery_max: # ensure that ev is not charging beyond capacity
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
		grid_nodes_load = self.get_grid_nodes_load(action)
		total_load = sum(grid_nodes_load)
		grid_nodes_load[0] = -total_load
		return grid_nodes_load

	def get_grid_nodes_load(self, action):
		load = np.zeros(self.grid.n_nodes)
		for ev_ind, charging_rate in enumerate(self.charge_action_to_list(action)):
			if charging_rate > 0:
				ev = self.fleet.vehicles[ev_ind]
				load[ev.grid_position] += ev.charge_rate * ev.power_consumption * charging_rate
		return load

	def print_transition_table(self):
		for action in self.get_actions():
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
				for s in self.get_states():
					actions = [self.charge_action_to_list(a) for a in policy[i][p_ind][s]]
					print("{:11}| {:5} | {:19}| {}".format(i, price, str(self.charge_state_to_list(s)), actions))
			print("")


		print("price | EVs Charging State | expected value")
		for p_ind, price in enumerate(self.get_prices(0)):
			for s in self.get_states():
				print("{:5} | {:19}| {}".format(price, str(self.charge_state_to_list(s)), expected_value[s][p_ind]))

	def solve_get_stats(self, simulation_repetions=300):
		initial_time = time.time()
		self.grid_feasible_actions()
		grid_feasible_actions_time = time.time() - initial_time

		initial_time = time.time()
		policy, expected_value = self.value_iteration()
		optimization_time = time.time() - initial_time

		simulation_results = self.run_simulations(policy=policy, initial_state=0, repetitions=simulation_repetions)

		return {"# Actions": len(self.get_actions()),
				"# Feasible actions": len(self.grid_feasible_actions()),
				"# States": len(self.get_states()),
				"Optimization time": optimization_time,
				"Feasible actions computational time": grid_feasible_actions_time,
				"Expected value initial state": expected_value[0][0],
				"average_reward" : simulation_results["average_reward"],
				"error" : simulation_results["error"],
				}


	def run_simulations(self, policy, initial_state=0, repetitions=100):
		total_rewards = list()
		for i in range(repetitions):
			results = self.run_simulation(initial_state, policy)
			total_rewards.append(results["accumulated_reward"][-1])

		return {"average_reward": np.average(total_rewards), "error": np.std(total_rewards)}

	def run_simulation(self, initial_state, policy):
		loads = list()
		total_loads = list()
		flows = list()
		rewards = list()
		accumulated_reward = list()
		prices = list()
		total_reward = 0

		current_state = initial_state
		price_ind = 0  # TODO choose a price randomly based on the price transition function
		current_price = self.get_prices(0)[price_ind]
		price_ind = 0
		for timestep in range(self.horizon):
			action = random.choice(policy[timestep][price_ind][current_state])
			reward = self.get_reward(current_state, action, current_price)

			load = self.get_load(action)
			loads.append(load)
			total_loads.append(-load[0])
			loads.append(self.get_load(action))
			flows.append(self.grid.compute_flow(load))
			rewards.append(reward)
			total_reward += reward
			accumulated_reward.append(total_reward)
			prices.append(current_price)

			next_state = self.get_next_state(current_state, action, timestep)
			next_price = self.get_next_price(current_price, timestep)
			next_price_ind = self.get_prices(timestep+1).index(next_price)
			# print(self.charge_state_to_list(current_state), current_price, self.charge_action_to_list(action), self.charge_state_to_list(next_state), reward, current_price)
			current_state = next_state
			current_price = next_price
			price_ind = next_price_ind
		return {"loads": loads,
				"total_loads": total_loads,
				"flows": flows,
				"rewards": rewards,
				"accumulated_reward": accumulated_reward,
				"prices": prices}

	def get_next_state(self, state, action, timestep):
		return choose(self.future_states_probabilities(from_state=state, action=action))

	def get_next_price(self, price, timestep):
		return choose(self.future_prices_and_probabilities(price, timestep))



class UncoordinatedMDP(MDP):

	def grid_feasible_actions(self):
		return self.get_actions()
################################################################################


def mdp_only_feasible_actions():
	evs = [EV(0, 3, 3, 1, grid_pos=1, deadline=23), EV(0, 4, 4, 1, grid_pos=2, deadline=23)]
	fleet = Fleet(evs)
	grid = Grid.load_grid_from_file('grids/grid_1.txt')
	mdp = MDP(fleet=fleet, grid=grid, horizon=6)
	return mdp


def mdp_with_unfeasible_actions():
	evs = [EV(0, 3, 3, 1, grid_pos=1, deadline=23), EV(0, 4, 4, 1, grid_pos=1, deadline=23)]
	fleet = Fleet(evs)
	grid = Grid.load_grid_from_file('grids/grid_1.txt')
	mdp = MDP(fleet=fleet, grid=grid, horizon=6)
	return mdp


def uncoordinated_mdp_with_unfeasible_actions():
	evs = [EV(0, 3, 3, 1, grid_pos=1, deadline=23), EV(0, 4, 4, 1, grid_pos=2, deadline=23)]
	fleet = Fleet(evs)
	grid = Grid.load_grid_from_file('grids/grid_1.txt')
	mdp = UncoordinatedMDP(fleet=fleet, grid=grid, horizon=6)
	return mdp


def test_state_to_list_to_state():
	mdp = mdp_only_feasible_actions()
	for state in range(mdp.num_states):
		print("state: %2d -> " % state, end='')
		csl = mdp.charge_state_to_list(state)
		print(csl, end=" -> ")
		print(mdp.charge_list_to_state(csl))


def test_action_to_list():
	mdp = mdp_only_feasible_actions()
	for action in range(mdp.num_actions):
		print("action: %d -> " % action, end='')
		print(mdp.charge_action_to_list(action))


def test_state_plus_action():
	mdp = mdp_only_feasible_actions()
	for state in range(mdp.num_states):
		for action in range(mdp.num_actions):
			print("state + action: %2d+%d -> " % (state, action), end='')
			csl = mdp.charge_list_apply_action_list(mdp.charge_state_to_list(state), mdp.charge_action_to_list(action))
			print(csl, end=" -> ")
			print(mdp.charge_list_to_state(csl))


	test_loads_feasibility(mdp_only_feasible_actions())


def test_loads_feasibility(mdp):
	for action in mdp.get_actions():
		print("action: ", action, mdp.charge_action_to_list(action))
		load = mdp.get_load(action=action)
		print("load: ", load)
		print("flow", mdp.grid.compute_flow(load))
		if mdp.grid.feasible(load):
			print("feasible")
		else:
			print("not feasible")
		print("")


def test_value_iteration():
	mdp_feasible = mdp_only_feasible_actions()
	policy_feasible, ev_feasible = mdp_feasible.value_iteration()
	mdp_feasible.print_policy_expected_value(policy_feasible, ev_feasible)

	mdp_unfeasible = mdp_with_unfeasible_actions()
	policy_unfeasible, ev_unfeasible = mdp_unfeasible.value_iteration()
	mdp_unfeasible.print_policy_expected_value(policy_unfeasible, ev_unfeasible)

	# Expected value in a grid with mdp with unfeasible actions
	# should always be equal or smaller than the mdp with only feasible actions
	for p_ind, price in enumerate(mdp_feasible.get_prices(0)):
		for s in mdp_feasible.get_states():
			assert ev_unfeasible[s][p_ind] <= ev_feasible[s][p_ind]


def test_coordinated_uncoordinated():
	mdp_coordinated_feasible = mdp_only_feasible_actions()
	policy_coordinated_feasible, ev_coordinated_feasible = mdp_coordinated_feasible.value_iteration()
	mdp_coordinated_feasible.print_policy_expected_value(policy_coordinated_feasible, ev_coordinated_feasible)

	mdp_uncoordinated = uncoordinated_mdp_with_unfeasible_actions()
	policy_uncoordinated, ev_uncoordinated = mdp_uncoordinated.value_iteration()
	mdp_uncoordinated.print_policy_expected_value(policy_uncoordinated, ev_uncoordinated)

	# Expected value in a grid with mdp with feasible actions and the uncoordinated mdp
	# should always be equal
	for p_ind, price in enumerate(mdp_coordinated_feasible.get_prices(0)):
		for s in mdp_coordinated_feasible.get_states():
			assert ev_coordinated_feasible[s][p_ind] == ev_uncoordinated[s][p_ind]

def test_simulations(mdp):
	policy, expected_value = mdp.value_iteration()
	print("expected_value: " + str(expected_value[0][0]))
	results = mdp.run_simulations(policy=policy, initial_state=0, repetitions=300)
	for key, result in results.items():
		print(key + ": " + str(result))

if __name__ == "__main__":
	test_state_to_list_to_state()
	test_action_to_list()
	test_state_plus_action()
	test_loads_feasibility(mdp_with_unfeasible_actions())
	test_loads_feasibility(mdp_only_feasible_actions())
	test_value_iteration()
	test_coordinated_uncoordinated()
	test_simulations(mdp_only_feasible_actions())
	test_simulations(mdp_with_unfeasible_actions())
	print(mdp_only_feasible_actions().solve_get_stats())
	print(mdp_with_unfeasible_actions().solve_get_stats())
	print(uncoordinated_mdp_with_unfeasible_actions().solve_get_stats())

