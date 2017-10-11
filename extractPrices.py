import numpy as np
import matplotlib.pyplot as plt

try: 
   prices = np.load('extractedPrices.npy')
   #print('prices loaded')
except IOError as e:
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
# print(prices[0:,36])
   np.save('extractedPrices', prices)

# processing data
minimumPerDay = np.zeros((360))
maximumPerDay = np.zeros((360))
for i in range(0,360):
   minimumPerDay[i] = min(prices[:,i])
   maximumPerDay[i] = max(prices[:,i])
#print(minimumPerDay)
#print(maximumPerDay)
averageMinimum = sum(minimumPerDay)/360
absoluteMinimum = min(minimumPerDay[:])
averageMaximum = sum(maximumPerDay)/360
absoluteMaximum = max(maximumPerDay[:])
# print(averageMinimum) # 22.7254
# print(absoluteMinimum) #-10.69 # on 5-8
# print(averageMaximum) # 57.711
# print(absoluteMaximum) # 874.01 # on 11-7

# see distribution of prices 
# generally expect prices between 0 and 100
minPrice = 0
maxPrice = 100
numberOfBuckets = 5 # specify your number of price levels
buckets = np.zeros((numberOfBuckets))
threshholds = []
for i in range (1,numberOfBuckets):
   threshholds.append(minPrice + i*(maxPrice/numberOfBuckets))
threshholds.append(maxPrice)
#print(threshholds)

sortedPrices = np.sort(prices, axis=None)
#print(sortedPrices)
i = 0
for t in range(0,numberOfBuckets-1):
   while(sortedPrices[i] < threshholds[t]):
      buckets[t] +=1
      i += 1
buckets[numberOfBuckets-1] += (len(sortedPrices) - i)
   
#print(buckets)
x = range(numberOfBuckets)
plt.bar(threshholds, buckets, color="blue")
plt.show()
    
priceSequence = prices.flatten(1)
