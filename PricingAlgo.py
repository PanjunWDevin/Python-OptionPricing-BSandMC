import math as mt
import scipy as sp
from scipy.stats import norm
import time
import numpy as np
from matplotlib import pylab
import matplotlib.pyplot as plt

# parameters
S_underline = 2.45
strike = 2.50
maturity = 0.25 #1/4 year
r= 0.05 # 5%
vol = 0.25 # volatility

def european_call_option(spot, strike, maturity, r, vol):

    d1 = (mt.log(spot/strike)+(r+0.5*vol*vol)*maturity)/(vol*mt.sqrt(maturity))
    d2 = d1 - vol*mt.sqrt(maturity)

    price = spot * norm.cdf(d1) - strike * mt.exp(-r*maturity) * norm.cdf(d2)
    return price

test_price = european_call_option(S_underline,strike,maturity,r,vol)
print test_price

portfolioSize = range(1, 10000, 500) #output a list with start at 1 and step 500 end at 10000
timeSpent = []

for size in portfolioSize:
    now = time.time() #timer
    strikes = np.linspace(2.0,3.0,size) #call numpy function linespace to create a array
    for i in range(size):   #test
        res = european_call_option(S_underline, strikes[i], maturity, r, vol)
    timeSpent.append(time.time() - now)

pylab.figure(figsize = (12,8))
pylab.bar(portfolioSize, timeSpent, color = 'r', width =300)
pylab.grid(True)
pylab.show()


# use numpy vector computation, reduce for loop usage
def european_call_option_numpy(spot, strike, maturity, r, vol):

    d1 = (np.log(spot/strike)+(r+0.5*vol*vol)*maturity)/(vol*np.sqrt(maturity))
    d2 = d1 - vol*np.sqrt(maturity)

    price = spot * norm.cdf(d1) - strike * np.exp(-r*maturity) * norm.cdf(d2)
    return price

timeSpentNumpy = []

for size in portfolioSize:
   now = time.time() #timer
    strikes = np.linspace(2.0,3.0,size) #call numpy function linespace to create a array
    res_temp = european_call_option_numpy(S_underline, strikes, maturity, r, vol)
    timeSpentNumpy.append(time.time() - now)

pylab.figure(figsize = (12,8))
pylab.bar(portfolioSize, timeSpentNumpy, color = 'blue', width =300)
pylab.grid(True)
pylab.show()

# Monte Carlo Simulation scipy monte carlo

import scipy

pylab.figure(figsize = (12,8))
randomSeries = scipy.random.randn(1000)
pylab.plot(randomSeries)
pylab.show()
print randomSeries.mean()
print randomSeries.std()

numOfpath = 5000 # MC paths

def call_option_pricer_MC(spot, strike, maturity, r, vol, numOfPath):
    randomSeries = scipy.random.randn(numOfPath)
    s_t = spot * np.exp((r - 0.5 * vol * vol) * maturity + randomSeries * vol * mt.sqrt(maturity))
    sumValue = np.maximum(s_t - strike, 0.0).sum()
    price = mt.exp(-r*maturity) * sumValue / numOfPath
    return price

# parameters
S_underline = 2.45
strike = 2.50
maturity = 0.25 #1/4 year
r= 0.05 # 5%
vol = 0.25 # volatility


price_MC = call_option_pricer_MC(S_underline,strike,maturity,r,vol,numOfpath)

#print price_MC

pathScenario = range(1000, 50000, 1000)
numberOfTrials = 100

confidenceIntervalUpper = []
confidenceIntervalLower = []
means = []

for scenario in pathScenario:
    res = np.zeros(numberOfTrials)
    for i in range(numberOfTrials):
        res[i] = call_option_pricer_MC(S_underline, strike, maturity, r, vol, numOfPath = scenario)
    means.append(res.mean())
    confidenceIntervalUpper.append(res.mean() + 1.96*res.std()) #95% confidence interval
    confidenceIntervalLower.append(res.mean() - 1.96*res.std()) #95% confidence interval

pylab.figure(figsize = (12,8))
price_MCs = np.array([means,confidenceIntervalUpper,confidenceIntervalLower]).T
pylab.plot(pathScenario, price_MCs)
pylab.show()

#also can calculate implied volatility
rom
scipy.optimize
import brentq


# Target Function
class cost_function:
    def __init__(self, target):
        self.targetValue = target

    def __call__(self, x):
        return call_option_pricer(spot, strike, maturity, r, x) - self.targetValue


# Assume using intial volatility 
target = call_option_pricer(spot, strike, maturity, r, vol)
cost_sampel = cost_function(target)

# Using brent functiont 
impliedVol = brentq(cost_sampel, 0.01, 0.5)

print 'implied volatility： %.2f' % (vol * 100,) + '%'
print 'implied volatility： %.2f' % (impliedVol * 100,) + '%'
