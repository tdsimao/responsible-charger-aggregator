import numpy as np
import matplotlib.pyplot as plt

def loadPrices():
	try: 
		prices = np.load('extractedPrices.npy')
		#print('prices loaded')
	except IOError as e:
		prices = loadFromFile()
	return prices

def loadFromFile():
	#days per month
	endDay = [31, 29, 31, 30 , 31, 30, 31, 31, 30, 31, 30, 25] 
	prices = np.zeros((24,360))

	# reading in the files
	day = 0
	for m in range(1, 13):
		for i in range (1, endDay[m-1]+1):
	 		date = str(m)+'-'+str(i) 		
	 		data = np.genfromtxt(date, delimiter=None, dtype = 'float')
	 		for h in range(0,24):
	    			prices[h][day] = data[h]
	 		day += 1
	np.save('extractedPrices', prices)
	return prices

def findMinMax(prices):
	minimumPerDay = np.zeros((360))
	maximumPerDay = np.zeros((360))
	for i in range(0,360):
		minimumPerDay[i] = min(prices[:,i])
		maximumPerDay[i] = max(prices[:,i])
	# print(minimumPerDay)
	# print(maximumPerDay)
	averageMinimum = sum(minimumPerDay)/360
	absoluteMinimum = min(minimumPerDay[:])
	averageMaximum = sum(maximumPerDay)/360
	absoluteMaximum = max(maximumPerDay[:])
	# print(averageMinimum) # 22.7254
	# print(absoluteMinimum) #-10.69 # on 5-8
	# print(averageMaximum) # 57.711
	# print(absoluteMaximum) # 874.01 # on 11-7
# remove outliers?

def splitAndVisualize (numberOfBuckets, prices, minPriceParam = 20, maxPriceParam = 60, thresholds=None):
	# plot distribution of prices 
	# generally expect prices between 0 and 100
	# optionally set manually threshholds like = [20, 30, 40, 50, 60, 70]

	nrBuckets = numberOfBuckets
	minPrice = minPriceParam
	maxPrice = maxPriceParam

	# if not given: compute threshholds
	if thresholds is None:
		thresholds = []
		for i in range (0,nrBuckets-1):
			thresholds.append(minPrice + i*(maxPrice/nrBuckets))
		thresholds.append(maxPrice)

	# sort into buckets and count
	buckets = np.zeros((nrBuckets))
	sortedPrices = np.sort(prices, axis=None)
	i = 0
	for t in range(0,nrBuckets-1):
		while(sortedPrices[i] < thresholds[t]):
			buckets[t] +=1
			i += 1
	buckets[nrBuckets-1] += (len(sortedPrices) - i) # rest of prices are above last threshold

	# plot distribution
	x = range(numberOfBuckets)
	plt.bar(thresholds, buckets, color="blue")
	plt.show()

	return thresholds
    
def pricesToSequence(priceArray):
	priceSequence = priceArray.flatten(1)
	priceseq = np.nan_to_num(priceSequence)
	return priceseq

def computeAverage(pricesequence):
	priceAvg = sum(pricesequence)/len(pricesequence)
	return priceAvg # 36.5

def randomAroundAvg(average): # optional scale/ size
	# get a random distribution around average
	deterministicAvg = np.random.normal(loc=average, scale=15, size=24)
	return deterministicAvg

def pricesToPriceLevels(numberOfLevels, prices, thresholds):
	try:
		priceLevelTable = np.load('level'+str(numberOfLevels)+'Prices.npy')
		# print("price level table loaded")
	except IOError as e:
   		priceLevelTable = np.zeros_like(prices)
   		for a in range (0,len(prices)):
      			for b in range (0, len(prices[0])):
         			t = 0
         			while((t < numberOfLevels -1) and (prices[a,b] > thresholds[t])):
            				t += 1
         			priceLevelTable[a,b] = t
   		np.save('level'+str(numberOfLevels)+'Prices', priceLevelTable)
	return priceLevelTable
 
def getPriceTransitionsTimeIndependent(nrLevels):
	# time independent price counts (Markov Chain)

	prices = loadPrices()
	thresholds = splitAndVisualize (nrLevels, prices)
	priceLevelTable = pricesToPriceLevels(nrLevels, prices, thresholds)
	priceLevelTable = priceLevelTable.astype(int) # 24x360

	# count level transitions
	transitionsCountMC = np.zeros((nrLevels, nrLevels))
	priceSequenceLevels = priceLevelTable.flatten(1)
	priceSequenceLevels = priceSequenceLevels.astype(int)

	for a in range (0, len(priceSequenceLevels) - 1):
		fromLevel = priceSequenceLevels[a]
		toLevel = priceSequenceLevels[a+1]
		transitionsCountMC[fromLevel, toLevel] += 1

	#calculate probabilities
	for i in range (0, nrLevels):
		summed = sum(transitionsCountMC[i,:])
		if(summed!= 0):
			for j in range (0, nrLevels):
				if(transitionsCountMC[i,j]/summed >= 0.01):
					transitionsCountMC[i,j] /= summed
				else:
					transitionsCountMC[i,j] = 0
		else:
			transitionsCountMC[i,:] = 0
	np.save('level'+str(nrLevels)+'PercentagesWithoutTime', transitionsCountMC)
	return transitionsCountMC

def getPriceTransitionsTimeDependent(nrLevels):
	# time dependent price discretization for each time step
	# time step distribution (24 distr.) - probs to get to other price level
	transitionsCount = np.zeros((24, nrLevels, nrLevels))

	prices = loadPrices()
	thresholds = splitAndVisualize (nrLevels, prices)
	priceLevelTable = pricesToPriceLevels(nrLevels, prices, thresholds)
	priceLevelTable = priceLevelTable.astype(int) # 24x360

	for hour in range (0, 24):
		for day in range (0, len(priceLevelTable[0])):
			fromLevel = priceLevelTable[hour, day]
			if (hour+1 == 24):
				toLevel = priceLevelTable[0,((day+1) % 360)]
			else: #(a+1 < 24)
				toLevel = priceLevelTable[(hour+1),day]
			transitionsCount[hour, toLevel, fromLevel] += 1
		for i in range (0, len(transitionsCount[1])):
			if(sum(transitionsCount[hour,i,:]) != 0):
				transitionsCount[hour,i,:] /= sum(transitionsCount[hour,i,:])			
			else:
				transitionsCount[hour,i,:] = 0
	np.save('level'+str(nrLevels)+'Percentages', transitionsCount)
	return transitionsCount

t1 = getPriceTransitionsTimeIndependent(4) 
# np.set_printoptions(precision=2)
# print(t1)

t2 = getPriceTransitionsTimeDependent(4)
# np.set_printoptions(precision=2)
# print(t2)
