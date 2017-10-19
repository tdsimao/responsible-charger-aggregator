from math import ceil

class EV:
	def __init__(self, battLevel, battMax, battGoal, chargeRate, gridPos, deadline):
		self.batteryLevel = battLevel	# The current battery level
		self.batteryMax = battMax		# The maximum battery level
		self.batteryGoal = battGoal		# The minimum charge required before the deadline
		self.chargeRate = chargeRate	# Charge rate of battery
		self.gridPosition = gridPos		# The node where this EV is charging on the grid
		self.chargeDeadline = deadline	# The time deadline which the EV must be chage by
		self.nChargeSteps = self.numTimestepsToChargeTotal()

	def isFullyCharged(self):
		return self.batteryLevel >= self.batteryMax

	def isGoalCharged(self):
		return self.batteryLevel >= self.batteryGoal

	def numTimestepsToChargeTotal(self):
		return int(ceil(self.batteryMax / self.chargeRate) + 1)

	def numTimestepsToChargeToFull(self):
		return int(ceil((self.batteryMax - self.batteryLevel) / self.chargeRate))

	def numTimestepsToChargeToGoal(self):
		return int(ceil((self.batteryGoal - self.batteryLevel) / self.chargeRate))

	def __str__(self):
		return "EV @: %2d, battery: %3d/%3d, charge rate: %3d" % (self.gridPosition, self.batteryLevel, self.batteryMax, self.chargeRate)

