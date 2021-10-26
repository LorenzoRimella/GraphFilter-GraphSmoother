import numpy as np
import random
import time
import networkx as nx
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt
import math

#########################################
#########################################


def extractprob(pi, T, nlayers, nstates, marginalise = True, partition= None):
    
    if marginalise:
    
        timeprob = list()
    
        for t in range(0,T):
        
            prob = np.zeros((nlayers, nstates))
        
            for K in range(0,len(partition)):
            
                for k in range(0, len(pi[t][K])):
                    current = findvalue(nstates, len(partition[K]) ,k)
                    for i in range(0,len(current)):
                        prob[int(partition[K][i][1])][current[i]] = prob[int(partition[K][i][1])][current[i]] + pi[t][K][k]
        
            timeprob.append(prob)
            
    else:
        timeprob = list()
        
        for t in range(0,T):
            prob = np.array([float(1)])
            for index in range(len(partition)):
        
                support2= prob[0]* pi[t][index]

                for i in range(1, np.size(prob)):
                    support1 = prob[i]* pi[t][index]
                    support2 = np.hstack((support2, support1))
                prob = support2
            
            timeprob.append(prob)
            
    return timeprob      

#########################################
#########################################
def prediction(pi, T, nlayers, nstates, marginalise = True, partition = None, rev = False):
    
    if rev==False:
        if marginalise:
            pred = np.zeros((T,nlayers))
        
            for t in range(0,T):
                prob = np.array([float(1)])
                for index in range(len(partition)):
        
                    support2 = prob[0]* pi[t][index]

                    for i in range(1, np.size(prob)):
                        support1 = prob[i]* pi[t][index]
                        support2 = np.hstack((support2, support1))
                    prob = support2
            
                k = np.argmax(prob)
            
                pred[t,:] = findvalue(nstates, nlayers, k)
        
        else:
            pred = np.zeros((T,nlayers))
            for t in range(0,T):
                k = np.argmax(pi[t])
                pred[t,:] = findvalue(nstates, nlayers, k)
            
            
            
                
    
    else:
        if marginalise:
            pred = np.zeros((T,nlayers))
        
            for t in reversed(range(0,T)):
                print(t)
                prob = np.array([float(1)])
                for index in range(len(partition)):
        
                    support2= prob[0]* pi[t][index]

                    for i in range(1, np.size(prob)):
                        support1 = prob[i]* pi[t][index]
                        support2 = np.hstack((support2, support1))
                    prob = support2
            
                k = np.argmax(prob)
            
                pred[T-t-1,:] = findvalue(nstates, nlayers, k)
        
                
        else:
            pred = np.zeros((T,nlayers))
            for t in reversed(range(0,T)):
                k = np.argmax(pi[t])
                pred[T-t-1,:] = findvalue(nstates, nlayers, k)
        
            
    return pred
