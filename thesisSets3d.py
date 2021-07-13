#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 01:14:55 2021

@author: axelronold
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import minimize
from tqdm import tqdm
from numba import njit
import utilities as ut

@njit
def funcSum2d(x,*args):
    return -(x[0]*args[0]+x[1]*args[1])

@njit
def funcSum3d(x,*args):
    return -(x[0]*args[0]+x[1]*args[1]+x[2]*args[2])

@njit
def funcConstX(x,*args):
    return -x[0]*args[0]

@njit
def funcConstY(x,*args):
    return -x[1]*args[1]

@njit
def funcConstZ(x,*args):
    return -x[2]*args[2]

@njit
def func_derivSum2d(x,*args):
    dfx0 = -args[0]
    dfx1 = -args[1]
    return np.array([dfx0,dfx1])

@njit
def func_derivSum3d(x,*args):
    dfx0 = -args[0]
    dfx1 = -args[1]
    dfx2 = -args[2]
    return np.array([dfx0,dfx1,dfx2])

@njit
def func_derivConstX2d(x,*args):
    dfx0 = -args[0]
    return np.array([dfx0,0])

@njit
def func_derivConstY2d(x,*args):
    dfx1 = -args[1]
    return np.array([0,dfx1])

@njit
def func_derivConstX3d(x,*args):
    dfx0 = -args[0]
    return np.array([dfx0,0,0])

@njit
def func_derivConstY3d(x,*args):
    dfx1 = -args[1]
    return np.array([0,dfx1,0])

@njit
def func_derivConstZ3d(x,*args):
    dfx2 = -args[2]
    return np.array([0,0,dfx2])

def optimizeConst(sets,wgt,i):
    bnds = ((sets[i][0], sets[i][0]+sets[i][-1]), (sets[i][1], sets[i][1]+sets[i][-1]),(sets[i][2], sets[i][2]+sets[i][-1]))
    if i==0:
        consX = ({'type': 'eq','fun' : lambda x: np.array([(x[0]-sets[i][0])**2+(x[1]-sets[i][1])**2+(x[2]-sets[i][2])**2-sets[i][-1]**2]),'jac' : lambda x: np.array([2*(x[0]-sets[i][0]), 2*(x[1]-sets[i][1]),2*(x[2]-sets[i][2])])},
                      {'type': 'ineq','fun' : lambda x: np.array([x[0]*wgt[0]-x[1]*wgt[1]]),'jac' : lambda x: np.array([wgt[0], -wgt[1], 0])},
                      {'type': 'ineq','fun' : lambda x: np.array([x[0]*wgt[0]-x[2]*wgt[2]]),'jac' : lambda x: np.array([wgt[0], 0, -wgt[2]])})
                    
        resX = minimize(funcConstX, [sets[i][0],sets[i][1],sets[i][2]+sets[i][-1]], args=(wgt[0],wgt[1],wgt[2],), jac=func_derivConstX3d,constraints=consX, method='SLSQP', bounds = bnds, options={'disp': False})
        return resX.x[0]
    
    elif i==1:            
        consY = ({'type': 'eq','fun' : lambda x: np.array([(x[0]-sets[i][0])**2+(x[1]-sets[i][1])**2+(x[2]-sets[i][2])**2-sets[i][-1]**2]),'jac' : lambda x: np.array([2*(x[0]-sets[i][0]), 2*(x[1]-sets[i][1]),2*(x[2]-sets[i][2])])},
                  {'type': 'ineq','fun' : lambda x: np.array([x[1]*wgt[1]-x[0]*wgt[0]]),'jac' : lambda x: np.array([-wgt[0], wgt[1], 0])},
                  {'type': 'ineq','fun' : lambda x: np.array([x[1]*wgt[1]-x[2]*wgt[2]]),'jac' : lambda x: np.array([0, wgt[1], -wgt[2]])})
                    
        resY = minimize(funcConstY, [sets[i][0]+sets[i][-1],sets[i][1],sets[i][2]], args=(wgt[0],wgt[1],wgt[2],), jac=func_derivConstY3d,constraints=consY, method='SLSQP', bounds = bnds, options={'disp': False})
        return resY.x[1]
    
    elif i==2:            
        consZ = ({'type': 'eq','fun' : lambda x: np.array([(x[0]-sets[i][0])**2+(x[1]-sets[i][1])**2+(x[2]-sets[i][2])**2-sets[i][-1]**2]),'jac' : lambda x: np.array([2*(x[0]-sets[i][0]), 2*(x[1]-sets[i][1]),2*(x[2]-sets[i][2])])},
                  {'type': 'ineq','fun' : lambda x: np.array([x[2]*wgt[2]-x[0]*wgt[0]]),'jac' : lambda x: np.array([-wgt[0], 0, wgt[2]])},
                  {'type': 'ineq','fun' : lambda x: np.array([x[2]*wgt[2]-x[1]*wgt[1]]),'jac' : lambda x: np.array([0, -wgt[1], wgt[2]])})
                    
        resZ = minimize(funcConstZ, [sets[i][0],sets[i][1]+sets[i][-1],sets[i][2]], args=(wgt[0],wgt[1],wgt[2],), jac=func_derivConstZ3d,constraints=consZ, method='SLSQP', bounds = bnds, options={'disp': False})
        return resZ.x[2]
    
def optimizeSum(sets,wgt,i):
    cons = ({'type': 'eq','fun' : lambda x: np.array([(x[0]-sets[i][0])**2+(x[1]-sets[i][1])**2+(x[2]-sets[i][2])**2-sets[i][-1]**2]),'jac' : lambda x: np.array([2*(x[0]-sets[i][0]), 2*(x[1]-sets[i][1]),2*(x[2]-sets[i][2])])})
            
    bnds = ((sets[i][0], sets[i][0]+sets[i][-1]), (sets[i][1], sets[i][1]+sets[i][-1]),(sets[i][2], sets[i][2]+sets[i][-1]))
                        
    res = minimize(funcSum3d, [sets[i][0],sets[i][1]+sets[i][-1],sets[i][2]], args=(wgt[0],wgt[1],wgt[2],), jac=func_derivSum3d,constraints=cons, method='SLSQP', bounds = bnds, options={'disp': False})
        
    return res.fun
        
#@njit
def wgtConst(sets,n = 2**6,convex=True):
    setsUpper = set()
    
    for i in range(int((n+1)*(n+2)/2)):
        if i==0:
            wgt = np.array([0.0,0.0,1.0])
        else:
            if wgt[-1] > 10**(-10):
                wgt = np.array([wgt[0],wgt[1]+1/n,wgt[2]-1/n])
            else:
                wgt[0] += 1/n
                wgt = np.array([wgt[0],0.0,1-wgt[0]])
        lowObj = np.full(len(wgt),np.inf)
        lowSetObj = np.full(len(wgt),-1.0)
        for i in range(len(sets)):
            currLowObj = np.full(len(wgt),np.inf)
            ends = np.zeros(len(wgt))
            
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

#@njit
def wgtSum(sets,n = 2**3,convex=True):
    if convex:
        setsUpper = set()

        for i in range(int((n+1)*(n+2)/2)):
            if i==0:
                wgt = np.array([0.0,0.0,1.0])
            else:
                if wgt[-1] > 10**(-10):
                    wgt = np.array([wgt[0],wgt[1]+1/n,wgt[2]-1/n])
                else:
                    wgt[0] += 1/n
                    wgt = np.array([wgt[0],0.0,1-wgt[0]])
            low = np.inf
            lowSet = -1
            for i in range(len(sets)):
                        
                value = -optimizeSum(sets,wgt,i)
                                   
                if value < low:
                    low = value
                    lowSet = i
            
            if lowSet != -1:
                setsUpper.add(lowSet)   
    
    return sorted(setsUpper)

def createAndPlot(nmbSolutions,nmbObjectives,nmbWeights=2**6,convex=True,plotting=False):    
    sets = np.random.rand(nmbSolutions,nmbObjectives+1)

    sets[:,:-1] = 3*sets[:,:-1]+1
    sets[:,-1] = 0.5*sets[:,-1]+0.25
     
    if plotting and nmbObjectives == 2:
        ut.plot2d(sets)
    elif plotting and nmbObjectives == 3:
        ut.plot3d(sets)
    
    if convex:
        start_timeSum = time.time()
        wgtSumfasit = wgtSum(sets,nmbWeights)
        totalTimeSum = round(time.time() - start_timeSum,4)
        
        start_timeConst = time.time()
        wgtConstfasit = wgtConst(sets,nmbWeights)
        totalTimeConst = round(time.time() - start_timeConst,4)
         
       #emergencySumNO = np.setdiff1d(wgtSumfasit,fasit)
       #emergencyConst = np.setdiff1d(wgtConstfasit,fasit)
    
    
    #else:
    #    for i in range(1,nmbSolutions+1):
    #        sets[i] = {'x': [round(3*np.random.rand()+1,2)],'y': [round(3*np.random.rand()+1,2)],'r': [round(0.6*np.random.rand()+0.3,2)]}
    #        sets[i]['x'].append(round(sets[i]['x'][0]+(-1)**(i)*(0.4*sets[i]['r'][0]*np.random.rand()+0.8),2))
    #        sets[i]['y'].append(round(sets[i]['y'][0]+(-1)**(i)*(0.4*sets[i]['r'][0]*np.random.rand()+0.8),2))
    #        d=np.sqrt((sets[i]['x'][1]-sets[i]['x'][0])**2 + (sets[i]['y'][1]-sets[i]['y'][0])**2)
    #        sets[i]['r'].append(round((2*sets[i]['r'][0])*np.random.rand()+(d-sets[i]['r'][0]),2))
    #    findIntersectAndAngleToPlot(sets)
        
    #    print('Weighted Sum Method found these sets to be Upper Set Less Order Efficient: ',wgtSum(sets,convex))
    
    return [len(wgtSumfasit),len(wgtConstfasit),len(set(wgtSumfasit+wgtConstfasit)),totalTimeSum,totalTimeConst]
    
    
if __name__ == "__main__":
    start_time = time.time()
    instances = 1
    nmbObjectives = 3
    
    wgtSumplotA = []
    wgtConstplotA = []
    foundUpperA = []
    nmb = []
    timeSum = []
    timeConst = []
    
    for nmbWeights in range(2,8):
        nmb.append(2**(nmbWeights))
        wgtSumAgg = 0
        wgtConstAgg = 0
        foundCombined = 0
        timeUsedSum = 0
        timeUsedConst = 0
        
        for _ in tqdm(range(instances)):
            stats = createAndPlot(10,nmbObjectives,2**(nmbWeights))
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
        
        print('wgtConst: ',wgtConstplotA,'wgtSum: ',wgtSumplotA,'combined: ',foundUpperA)
        
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
        
    print("\n-----------------------------------------------------------")
    
    print('wgtConst: ',wgtConstplotA,'wgtSum: ',wgtSumplotA,'combined: ',foundUpperA)
    
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