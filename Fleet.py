class Fleet:
	def __init__(self, vehicles=None):
		if vehicles is None:
			self.vehicles = list()
		else:
			self.vehicles = vehicles

	def add_vehicle(self, vehicle):
		self.vehicles.append(vehicle)

	def size(self):
		return len(self.vehicles)

	def __str__(self):
		result = ""
		for v in self.vehicles:
			result += str(v) + "\n"
		return result
