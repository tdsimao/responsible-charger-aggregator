from EV import EV

class Fleet:
	def __init__(self, vehicles=[]):
		self.vehicles = vehicles
		# constructor

	def addVehicle(self, vehicle):
		self.vehicles.append(vehicle)

	def size(self):
		return len(self.vehicles)

	def __str__(self):
		result = ""
		for i in range(self.size()):
			result += str(self.vehicles[i]) + "\n"
		return result


