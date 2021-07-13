#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 14:28:52 2021

@author: axelronold
""" 

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import minimize
from tqdm import tqdm
from matplotlib.pyplot import cm

#the first six functions are only defining the objective function and the derivative
#This is what the scipy.optimize library needs to run
def funcSum(x,*args):
    return -(x[0]*args[0]+x[1]*args[1])

def funcConstX(x,*args):
    return -x[0]*args[0]

def funcConstY(x,*args):
    return -x[1]*args[1]

def func_derivSum(x,*args):
    dfx0 = -args[0]
    dfx1 = -args[1]
    return np.array([dfx0,dfx1])

def func_derivConstX(x,*args):
    dfx0 = -args[0]
    return np.array([dfx0,0])

def func_derivConstY(x,*args):
    dfx1 = -args[1]
    return np.array([0,dfx1])

#optimization function for weighted constraint method. Using the scipy.optimize library
def optimizeConst(sets,wgt,i):
    if i==0:
        consX = ({'type': 'eq','fun' : lambda x: np.array([(x[0]-sets[i][0])**2+(x[1]-sets[i][1])**2-sets[i][2]**2]),'jac' : lambda x: np.array([2*(x[0]-sets[i][0]), 2*(x[1]-sets[i][1])])},
                      {'type': 'ineq','fun' : lambda x: np.array([x[0]*wgt[0]-x[1]*wgt[1]]),'jac' : lambda x: np.array([wgt[0], -wgt[1]])})
        
        bnds = ((sets[i][0], sets[i][0]+sets[i][2]), (sets[i][1], sets[i][1]+sets[i][2]))
                    
        resX = minimize(funcConstX, [sets[i][0],sets[i][1]+sets[i][2]], args=(wgt[0],wgt[1],), jac=func_derivConstX,constraints=consX, method='SLSQP', bounds = bnds, options={'disp': False})
        return resX.x[0]
    
    elif i==1:            
        consY = ({'type': 'eq','fun' : lambda x: np.array([(x[0]-sets[i][0])**2+(x[1]-sets[i][1])**2-sets[i][2]**2]),'jac' : lambda x: np.array([2*(x[0]-sets[i][0]), 2*(x[1]-sets[i][1])])},
                  {'type': 'ineq','fun' : lambda x: np.array([x[1]*wgt[1]-x[0]*wgt[0]]),'jac' : lambda x: np.array([-wgt[0], wgt[1]])})
        
        bnds = ((sets[i][0], sets[i][0]+sets[i][2]), (sets[i][1], sets[i][1]+sets[i][2]))
                    
        resY = minimize(funcConstY, [sets[i][0]+sets[i][2],sets[i][1]], args=(wgt[0],wgt[1],), jac=func_derivConstY,constraints=consY, method='SLSQP', bounds = bnds, options={'disp': False})
        return resY.x[1]

#optimization function for weighted sum method. Using the scipy.optimize library    
def optimizeSum(sets,wgt,i):
    cons = ({'type': 'eq','fun' : lambda x: np.array([(x[0]-sets[i][0])**2+(x[1]-sets[i][1])**2-sets[i][2]**2]),'jac' : lambda x: np.array([2*(x[0]-sets[i][0]), 2*(x[1]-sets[i][1])])})
        
    bnds = ((sets[i][0], sets[i][0]+sets[i][2]), (sets[i][1], sets[i][1]+sets[i][2]))
                    
    res = minimize(funcSum, [sets[i][0],sets[i][1]+sets[i][2]], args=(wgt[0],wgt[1],), jac=func_derivSum,constraints=cons, method='SLSQP', bounds = bnds, options={'disp': False})
    return res.fun

#only used to plot the problem instances
def plot(sets,convex=True):    
    fig, ax = plt.subplots(1, 1)
    color=cm.rainbow(np.linspace(0,1,len(sets)))
    for i in range(len(sets)):
        c = color[i]
        theta = np.linspace(0, 2*np.pi, 100)
    
        x1 = sets[i][0]+sets[i][-1]*np.cos(theta)
        x2 = sets[i][1]+sets[i][-1]*np.sin(theta)
    
        ax.plot(x1, x2,color=c)
        ax.annotate(i,(sets[i][0],sets[i][1]),color=c)
    ax.set_aspect('equal', adjustable='datalim')
    ax.plot()
    plt.axis('equal')
    plt.show()

#implementation of the weighted constraint method        
def wgtConst(sets,n = 2**11,convex=True):
    setsUpper = set()
    
    for i in range(0,n+1):
        #go through all the number of weights, works best when n is a power of 2
        #because of memory
        wgt = np.array([i/n,(n-i)/n])
        lowObj = np.full(len(wgt),np.inf)
        lowSetObj = np.full(len(wgt),-1.0)
        
        #go through every feasible element to find the minimum
        for i in range(len(sets)):
            currLowObj = np.full(len(wgt),np.inf)
            ends = np.zeros(len(wgt))
            
            #since all the feasible elements in our problem instance is a circle
            #we can do a check to see if the maximum is feasible, if not then we 
            #need to use scipy.optimize to find the max that is feasible if any point is feasible at all
            for j in range(len(wgt)):
                for k in range(len(wgt)):
                    if j != k and ends[j] != -1:
                        if wgt[j]*(sets[i][j]+sets[i][-1]) >= wgt[k]*sets[i][k]:
                            ends[j] += 1
                        elif wgt[j]*(sets[i][j]) < wgt[k]*(sets[i][k]+sets[i][-1]):
                            ends[j] = -1
                if ends[j] == len(wgt)-1:
                    currLowObj[j] = wgt[j]*(sets[i][j]+sets[i][-1])
                elif ends[j] != -1:
                    ends[j] = 0
                    currLowObj[j] = wgt[j]*optimizeConst(sets,wgt,j)
            
            for c in range(len(wgt)):
                if currLowObj[c] < lowObj[c]:
                    lowObj[c] = currLowObj[c]
                    lowSetObj[c] = i
        
        if lowSetObj[0] != -1 and len(set(lowSetObj))==1:
            setsUpper.add(lowSetObj[0])
     
    return sorted(setsUpper)

#implementation of the weighted sum method
def wgtSum(sets,n = 2**11):
    setsUpper = set()
    
    for i in range(0,n+1):
        wgt = np.array([i/n,(n-i)/n])
        low = np.inf
        lowSet = 0
        for i in range(len(sets)):
                    
            value = -optimizeSum(sets,wgt,i)
                               
            if value < low:
                low = value
                lowSet = i
        
        setsUpper.add(lowSet)   
    
    return sorted(setsUpper)

#here we generate a problem instance
def createProblemInstance(nmbSolutions,nmbObjectives,nmbWeights=2**2,convex=True,plotting=True):    
    sets = np.random.rand(nmbSolutions,nmbObjectives+1)

    sets[:,:-1] = 3*sets[:,:-1]+1
    sets[:,-1] = 0.5*sets[:,-1]+0.25
     
    if plotting:
        plot(sets)   
    
    #both methods are given the number of weights they are going to solve for    
    start_timeSum = time.time()
    wgtSumfasit = wgtSum(sets,nmbWeights)
    totalTimeSum = round(time.time() - start_timeSum,4)
        
    start_timeConst = time.time()
    wgtConstfasit = wgtConst(sets,nmbWeights)
    totalTimeConst = round(time.time() - start_timeConst,4)
        
    return [len(wgtSumfasit),len(wgtConstfasit),len(set(wgtSumfasit+wgtConstfasit)),totalTimeSum,totalTimeConst]
    
    
if __name__ == "__main__":
    start_time = time.time()
    
    #the parameters we can alter to change the problem instances
    instances = 1
    nmbObjectives = 2
    nmbSolutions = 10
    
    wgtSumplotA = []
    wgtConstplotA = []
    foundUpperA = []
    nmb = []
    timeSum = []
    timeConst = []
    
    #now my code is used for testing the number of weights and how that affects the methods
    for nmbWeights in range(2,16):
        nmb.append(2**(nmbWeights))
        wgtSumAgg = 0
        wgtConstAgg = 0
        foundCombined = 0
        timeUsedSum = 0
        timeUsedConst = 0
        
        for _ in tqdm(range(instances)):
            #the next line instigate the whole problem and report the stats back
            stats = createProblemInstance(nmbSolutions,nmbObjectives,2**(nmbWeights))
            wgtSumAgg += stats[0]
            wgtConstAgg += stats[1]
            foundCombined  += stats[2]
            timeUsedSum += stats[3]
            timeUsedConst += stats[4]

        wgtConstplotA.append(wgtConstAgg/instances)
        wgtSumplotA.append(wgtSumAgg/instances)
        foundUpperA.append(foundCombined/instances)
        timeSum.append(timeUsedSum/instances)
        timeConst.append(timeUsedConst/instances)
        
    print("\n-----------------------------------------------------------")
    
    print('wgtConst: ',wgtConstplotA,'wgtSum: ',wgtSumplotA,'combined: ',foundUpperA)
    
    #plot the result
    plt.plot(nmb,wgtConstplotA,color='blue',label='Wgt Constraint')
    plt.plot(nmb,wgtSumplotA,color='green',label='Wgt Sum')
    plt.plot(nmb,foundUpperA,color='black',label='Combined')
    plt.xlabel('Number of Weights')
    plt.ylabel('Number of solutions found to be u.s.l.o.e.')
    plt.xscale('log', basex=2)
    plt.legend()
    plt.show()
    
    plt.plot(nmb,timeConst,color='blue',label='Wgt Constraint')
    plt.plot(nmb,timeSum,color='green',label='Wgt Sum')
    plt.xlabel('Number of Weights')
    plt.ylabel('Time spent per instance [s]')
    plt.xscale('log', basex=2)
    plt.yscale('log')
    plt.legend()
    plt.show()
    
    print("--- %s seconds ---" % (round(time.time() - start_time,4)))