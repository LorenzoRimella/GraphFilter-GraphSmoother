import numpy as np
import random
import time
import math

import GraphFilter_GraphSmoother

    
#########################################
#########################################


def srGJ(mu0, P, h, T, epsilon= 2**(-1000)):
    #Filtering
    filtering = list()
    filtering.append(mu0)
    nstates = np.size(P,0)
    
    logfiltering = list()
    logfiltering.append(mu0 + epsilon)
    
    for t in range(1,T):
#         newfilter = np.dot(filtering[t-1], P)
        
#         for i in range(0, np.size(h,2)):
#             newfilter[i] = newfilter[i]*h[t,0, i]
            
#         filtering.append(newfilter/sum(newfilter))
        
        newlogfiltersum = np.zeros(nstates)
        if np.log(np.mean(np.exp(logfiltering[t-1])))<0:
            c = np.log(np.min(np.exp(logfiltering[t-1])))
            c = max(c, -700)
        else:
            c = np.log(np.max(np.exp(logfiltering[t-1])))
            c = min(c, 700)
            
        if c<=0:
            for i in range(nstates):
                    for j in range(nstates): 
                        newlogfiltersum[i]= newlogfiltersum[i] + np.exp( logfiltering[t-1][j]+ np.log(P[j,i])- c)
                        
                    newlogfiltersum[i]= np.log(newlogfiltersum[i]) + np.exp(c)
                    
        else:
            for i in range(nstates):
                    for j in range(nstates): 
                        newlogfiltersum[i]= newlogfiltersum[i] + np.exp( logfiltering[t-1][j]+ np.log(P[j,i])+ c)
                        
                    newlogfiltersum[i]= np.log(newlogfiltersum[i]) - np.exp(c)
        
                    
        for i in range(0, np.size(h,2)):
            newlogfiltersum[i] = newlogfiltersum[i]*np.log(h[t,0, i])
            
            
        
        if any(np.exp(newlogfiltersum)==0):
            filtering.append(np.array([[1.0,0.0],[0.0,1.0]], dtype=float)[np.where(newlogfiltersum==min(newlogfiltersum))][0])
            logfiltering.append(np.log(filtering[t]+epsilon))
        else:
            if any(newlogfiltersum>700):
                filtering.append(np.array([[1.0,0.0],[0.0,1.0]], dtype=float)[np.where(newlogfiltersum==max(newlogfiltersum))][0])
                logfiltering.append(np.log(filtering[t]+epsilon))            
            else:
                filtering.append(np.exp(newlogfiltersum)/sum(np.exp(newlogfiltersum)))
                logfiltering.append(newlogfiltersum)
        
    
    ##############################################################################################################
        
    smoothing = list()
    smoothing.append(filtering[T-1])
    
    logsmoothing = list()
    logsmoothing.append(logfiltering[T-1])
    
    
    for t in range(1,T):
        current = np.ones(np.size(P,0))
        
        if np.log(np.mean(np.exp(h[T-t,0,:]))) <0:
            c = np.log(np.min(np.exp(h[T-t,0,:])))
            c = max(c, -650)
        else:
            c = np.log(np.max(np.exp(h[T-t,0,:])))
            c = min(c,650)
        
        if c <= 0:
            for i in range(nstates):
                for j in range(nstates): 
                    current[i] = current[i] + np.exp(np.log(P[j,i]) + np.log(h[T-t,0,j])-c)
                
                if current[i] + np.exp(c)<0:
                    current[i] = np.log(epsilon)
                else:
                    current[i] = np.log( current[i] + np.exp(c) )
        
        else:
            for i in range(nstates):
                for j in range(nstates): 
                    current[i] = current[i] + np.exp(np.log(P[j,i]) + np.log(h[T-t,0,j])+c)

                if current[i] - np.exp(c)<0:
                    current[i] = np.log(epsilon)
                else:                    
                    current[i] = np.log( current[i] - np.exp(c) )        
                    
        newlogsmoothing = current+ logfiltering[T-t-1]
        
        
        if any(np.exp(newlogsmoothing)==0):
            smoothing.append(np.array([[1.0,0.0],[0.0,1.0]], dtype=float)[np.where(newlogsmoothing == min(newlogsmoothing))][0])
            logsmoothing.append(np.log(smoothing[t]+epsilon))
            
        else:    
            if any( (newlogsmoothing)>700):
                smoothing.append(np.array([[1.0,0.0],[0.0,1.0]], dtype=float)[np.where(newlogsmoothing == max(newlogsmoothing))][0])
                logsmoothing.append(np.log(smoothing[t]+epsilon))                
            else:
                logsmoothing.append(newlogsmoothing)
                smoothing.append(np.exp(newlogsmoothing)/sum(np.exp(newlogsmoothing)))
        
    return(smoothing)



    
#########################################
#########################################

def srGJ2(mu0, P, h, T):

	nstates = np.size(P,0)
	filtering = np.zeros((T,nstates),dtype= float)
	smoothing = np.zeros((T,nstates),dtype= float)     

	filtering[0,:]= mu0

	for t in range(1,T):
		normt = 0.0
		filtering[t,:]= h[t,:]*np.dot(filtering[t-1,:],P)
		filtering[t,:] = filtering[t,:]/sum(filtering[t,:])
            
	smoothing[T-1,:] = filtering[T-1,:]
	for t in range(1,T):
		layerjoint = np.zeros((nstates,nstates),dtype=float)
		for s1 in range(0,nstates):
			normjoint = 0

			for s2 in range(0,nstates):
				layerjoint[s2,s1] = P[s2,s1]*filtering[T-t-1,s2]*smoothing[T-t,s1]
				normjoint = normjoint+P[s2,s1]*filtering[T-t-1,s2]

			layerjoint[:,s1]= layerjoint[:,s1]/normjoint

		smoothing[T-t-1,:]= layerjoint.sum(axis=0)   

            
	return( smoothing )
