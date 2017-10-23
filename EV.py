from math import ceil

class EV:
	def __init__(self, batt_level, batt_max, batt_goal, charge_rate, grid_pos, deadline):
		self.battery_level = batt_level	# The current battery level
		self.battery_max = batt_max		# The maximum battery level
		self.battery_goal = batt_goal		# The minimum charge required before the deadline
		self.charge_rate = charge_rate	# Charge rate of battery
		self.grid_position = grid_pos		# The node where this EV is charging on the grid
		self.charge_deadline = deadline	# The time deadline which the EV must be chage by
		self.num_charge_steps = self.num_timesteps_to_charge_total()

	def is_fully_charged(self):
		return self.battery_level >= self.battery_max

	def is_goal_charged(self):
		return self.battery_level >= self.battery_goal

	def num_timesteps_to_charge_total(self):
		return int(ceil(self.battery_max / self.charge_rate) + 1)

	def num_timesteps_to_charge_to_full(self):
		return int(ceil((self.battery_max - self.battery_level) / self.charge_rate))

	def num_timesteps_to_charge_to_goal(self):
		return int(ceil((self.battery_max - self.battery_level) / self.charge_rate))

	def __str__(self):
		return "EV @: %2d, battery: %3d/%3d, charge rate: %3d" % (self.grid_position, self.battery_level, self.battery_max, self.charge_rate)

