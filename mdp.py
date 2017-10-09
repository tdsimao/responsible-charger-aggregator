# Vincent Koeten
# CS4010 HW 1
# September 2017

from pprint import pprint

nStates = 14
nActions = 4 # H0=0, H1=1, H2=2, MOVE=3
reward = []
transitionTable = []
discounts = [0.95, 0.9, 0.85, 0.8]

def MDP(discount):
	initializeTransitionTable()
	initializeRewardTable()
	optimalAction = solve(discount)
	return optimalAction

# Solve MDP and return optimal action
# @param discountFactor the discount factor
# @return optimal action to select in state s_0
def solve(discountFactor):
	assert discountFactor >= 0 and discountFactor < 1.0
	bestAction = -1
	qn = [[reward[s][a] for a in range(nActions)] for s in range(nStates)]
	maxDiff = 1
	# The value iteration algorithm
	while maxDiff > 0.001:
		qnp1 = [[0 for a in range(nActions)] for s in range(nStates)]
		for s in range(nStates):
			for a in getFeasibleActions(s):
				qnp1[s][a] = reward[s][a] + discountFactor * discountReward(qn, s, a)
		maxDiff = max(getDifferences(qn, qnp1))
		qn = qnp1
	bestAction = max(xrange(len(qn[0])), key=qn[0].__getitem__)
	return bestAction

def getDifferences(qn, qnp1):
	diff = set()
	for s in range(nStates):
		for a in range(nActions):
			diff.add(abs(qn[s][a] - qnp1[s][a]))
	return diff

def discountReward(qn, s, a):
	result = 0;
	for sp in range(nStates):
		prob = transitionTable[s][a][sp]
		if prob is not 0:
			m = max(qn[sp])
			result += prob * m
	return result

def getFeasibleActions(s):
	assert s>=0 and s<=13

	actions = []
	if s is 0:
		actions.append(0)
		actions.append(1)
		actions.append(2)
	else:
		actions.append(3)
	
	return actions


def initializeRewardTable():
	global reward
	reward = [[0 for a in range(nActions)] for s in range(nStates)]
	reward[4][3] = 100.0
	reward[8][3] = 105.0
	reward[11][3] = 52.0
	reward[12][3] = 53.0
	reward[0][0] = 100.0
	reward[0][1] = 100.0
	reward[0][2] = 100.0


def initializeTransitionTable():
	global transitionTable
	transitionTable = [[[0 for s in range(nStates)] for a in range(nActions)] for t in range(nStates)]
	
	# state 0 (feasible actions: 0 1 2)
	transitionTable[0][0][1] = 1.0
	transitionTable[0][1][5] = 1.0
	transitionTable[0][2][9] = 1.0
	
	# hallway 0 (feasible actions: 3)
	for s in range(1, 3):
		transitionTable[s][3][s+1] = 0.8
		transitionTable[s][3][s] = 0.2
	
	transitionTable[3][3][4] = 1.0
	transitionTable[4][3][13] = 1.0
	
	# hallway 1 (feasible actions: 3)
	for s in range(5, 7):
		transitionTable[s][3][s+1] = 0.6
		transitionTable[s][3][s] = 0.4
	
	transitionTable[7][3][8] = 1.0
	transitionTable[8][3][13] = 1.0
	
	# hallway 2 (feasible actions: 3)
	for s in range(9, 11):
		transitionTable[s][3][s+1] = 0.5
		transitionTable[s][3][s] = 0.5
	
	transitionTable[11][3][12] = 1.0
	transitionTable[12][3][13] = 1.0
	
	# goal state
	transitionTable[13][0][13] = 1.0
	transitionTable[13][1][13] = 1.0
	transitionTable[13][2][13] = 1.0
	transitionTable[13][3][13] = 1.0

def runDiscountRange(discounts):
	for discount in discounts:
		runDiscount(discount)

def runDiscount(discount):
	print "Best Action for discount", discount, ":", MDP(discount)

runDiscountRange([d*0.01 for d in range(100, 0, -1)])
# runDiscount(0.95)
