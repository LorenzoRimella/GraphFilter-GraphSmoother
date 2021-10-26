import numpy as np
import random
import time
import networkx as nx
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt
import math

import GraphFilter_GraphSmoother

#########################################
#########################################

def GJforward(mu0, transP, YT, W, C):
    nlayers = np.size(W, 0)
    nstates = np.size(W, 2)
    nobservations = np.size(YT, 1)
    T = np.size(YT, 0)
    
    # Forward
    alpha = list()
    filtering = list()
    
    newmu0 = mu0[0]

    for i in range(1,len(mu0)):
        supp1 = newmu0[0]*mu0[i]
            
        for j in range(1,len(newmu0)):
            supp2 = newmu0[j]*mu0[i]
            supp1 = np.hstack((supp1, supp2))
        newmu0 = supp1
        
    alpha.append(newmu0)
    filtering.append(newmu0)
    
    
    
    for t in range(1,T):
        newalpha1 = alpha[t-1]
        for m in reversed(range(0, nlayers)):
            index = list(range(0,nstates**nlayers))
            newalpha2 = np.zeros(nstates**nlayers)
            while(len(index)!=0):
                start = index[0]
                for i in range(0,nstates):
                    index.remove(start+ i*nstates**(nlayers-m-1))
                    for j in range(0,nstates):
                        newalpha2[start+ j*nstates**(nlayers-m-1)] = newalpha2[start+ j*nstates**(nlayers-m-1)]+ transP[m,i,j]* newalpha1[start+ i*nstates**(nlayers-m-1)]
                        
            newalpha1 = newalpha2
        
        sumalpha = 0
        for k in range(0,nstates**nlayers):
            newalpha1[k] = GraphFilter_GraphSmoother.Gtx(range(0,len(YT[t])), YT[t], C, W, GraphFilter_GraphSmoother.findvalue(nstates, nlayers, k), range(0,nlayers))*newalpha1[k]
            sumalpha = sumalpha+ newalpha1[k]
                            
        alpha.append(newalpha1/sumalpha)                 
        filtering.append(newalpha1/sumalpha)   
    
    
    return filtering


#########################################
#########################################

def GJforwardlog(mu0, transP, YT, W, C, epsilon= 2**(-1000)):
    nlayers = np.size(W, 0)
    nstates = np.size(W, 2)
    nobservations = np.size(YT, 1)
    T = np.size(YT, 0)
    
    # Forward
    logalpha = list()
    filtering = list()
    
    newmu0 = mu0[0]

    for i in range(1,len(mu0)):
        supp1 = newmu0[0]*mu0[i]
            
        for j in range(1,len(newmu0)):
            supp2 = newmu0[j]*mu0[i]
            supp1 = np.hstack((supp1, supp2))
        newmu0 = supp1

    alpha0 = newmu0+epsilon
    logalpha.append(np.log(alpha0))
    filtering.append(newmu0)
    
    
    
    for t in range(1,T):
            newlogalpha1 = logalpha[t-1]
            for m in reversed(range(0, nlayers)):
                c = np.log(np.mean(np.exp(newlogalpha1)))
                index = list(range(0,nstates**nlayers))
                newtransalpha2 = np.zeros(nstates**nlayers)
                while(len(index)!=0):
                    start = index[0]
                    for i in range(0,nstates):
                        index.remove(start+ i*nstates**(nlayers-m-1))
                        for j in range(0,nstates):
                            if c<=0:
                                if transP[m,i,j]==0:
                                    newtransalpha2[start+ j*nstates**(nlayers-m-1)] = newtransalpha2[start+ j*nstates**(nlayers-m-1)]+ np.exp(np.log(transP[m,i,j]+epsilon)+ (newlogalpha1[start+ i*nstates**(nlayers-m-1)])- c)         
                                else:
                                    newtransalpha2[start+ j*nstates**(nlayers-m-1)] = newtransalpha2[start+ j*nstates**(nlayers-m-1)]+ np.exp(np.log(transP[m,i,j])+ (newlogalpha1[start+ i*nstates**(nlayers-m-1)])- c) 
                                
                            else:
                                if transP[m,i,j]==0:
                                    newtransalpha2[start+ j*nstates**(nlayers-m-1)] = newtransalpha2[start+ j*nstates**(nlayers-m-1)]+ np.exp(np.log(transP[m,i,j]+epsilon)+ (newlogalpha1[start+ i*nstates**(nlayers-m-1)])+ c)         
                                else:
                                    newtransalpha2[start+ j*nstates**(nlayers-m-1)] = newtransalpha2[start+ j*nstates**(nlayers-m-1)]+ np.exp(np.log(transP[m,i,j])+ (newlogalpha1[start+ i*nstates**(nlayers-m-1)])+ c) 
                                
                                
                        
                if c<=0:        
                    newlogalpha1 = np.log(newtransalpha2)+ np.exp(c)
                else:
                    newlogalpha1 = np.log(newtransalpha2)- np.exp(c)
        
            sumalpha = 0
            for k in range(0,nstates**nlayers):
                newlogalpha1[k] = np.log(GraphFilter_GraphSmoother.Gtx(range(0,len(YT[t])), YT[t], C, W, GraphFilter_GraphSmoother.findvalue(nstates, nlayers, k), range(0,nlayers)))+newlogalpha1[k]
                            
            logalpha.append(newlogalpha1)        
            filt= np.exp(newlogalpha1)
            filtering.append(filt/sum(filt))   
    
    
    return {'logalpha':logalpha,'filtering':filtering}

#########################################
#########################################

def GJbackward(filtering, transP, YT, W, C):
    nlayers = np.size(W, 0)
    nstates = np.size(W, 2)
    nobservations = np.size(YT, 1)
    T = np.size(YT, 0)
    
    # Forward
    beta = list()
    smoothing = list()
    
    beta.append(np.ones(nstates**nlayers, dtype=float))
    smoothing.append(filtering[T-1])   
    
    
    for t in range(1,T-1):
        newbeta1 = beta[t-1]
        newbeta2 = np.zeros(nstates**nlayers)
        
        # M step
        for k in range(0,nstates**nlayers):
            newbeta2[k] = GraphFilter_GraphSmoother.Gtx(range(0,len(YT[T-t-1])), YT[T-t-1], C, W, GraphFilter_GraphSmoother.findvalue(nstates, nlayers, k), range(0,nlayers))*newbeta1[k]
        
        newbeta2 = newbeta2/sum(newbeta2)
            
        for m in reversed(range(0, nlayers)):
            index = list(range(0,nstates**nlayers))
            newbeta3 = np.zeros(nstates**nlayers)
            while(len(index)!=0):
                start = index[0]
                for i in range(0,nstates):
                    index.remove(start+ i*nstates**(nlayers-m-1))
                    for j in range(0,nstates):
                        newbeta3[start+ j*nstates**(nlayers-m-1)] = newbeta3[start+ j*nstates**(nlayers-m-1)]+ transP[m,j,i]* newbeta2[start+ i*nstates**(nlayers-m-1)]
                        
            newbeta2 = newbeta3
        
        newbeta1 = newbeta2
        beta.append(newbeta1/sum(newbeta1))
        
        smooth = filtering[T-t-1]*beta[t]
        smoothing.append(smooth/sum(smooth))
        
    smooth = filtering[0]
    smoothing.append(smooth/sum(smooth))
    
    return smoothing
#########################################
#########################################

def GJbackwardlog(logalpha, transP, YT, W, C, epsilon = 2**(-1000)):
    nlayers = np.size(W, 0)
    nstates = np.size(W, 2)
    nobservations = np.size(YT, 1)
    T = np.size(YT, 0)
    
    # Forward
    logbeta = list()
    smoothing = list()
    
    logbeta.append(np.log(np.ones(nstates**nlayers, dtype=float)))
    
    smooth = np.exp(logalpha[T-1-0]+logbeta[0]) 
    smoothing.append(smooth/sum(smooth))
    
    
    for t in range(1,T):
        newlogbeta1 = logbeta[t-1]
        newlogbeta2 = np.zeros(nstates**nlayers)
        
        # M step
        for k in range(0,nstates**nlayers):
            newlogbeta2[k] = np.log(GraphFilter_GraphSmoother.Gtx(range(0,len(YT[T-t])), YT[T-t], C, W, GraphFilter_GraphSmoother.findvalue(nstates, nlayers, k), range(0,nlayers)))+newlogbeta1[k]


        for m in reversed(range(0, nlayers)):
            c = np.log(np.mean(np.exp(newlogbeta2)))
            index = list(range(0,nstates**nlayers))
            newtransbeta3 = np.zeros(nstates**nlayers)
            while(len(index)!=0):
                start = index[0]
                for i in range(0,nstates):
                    index.remove(start+ i*nstates**(nlayers-m-1))
                    for j in range(0,nstates):
                            if c<=0:
                                if transP[m,j,i]==0:
                                    newtransbeta3[start+ j*nstates**(nlayers-m-1)] = newtransbeta3[start+ j*nstates**(nlayers-m-1)]+ np.exp(np.log(transP[m,j,i]+epsilon)+ newlogbeta2[start+ i*nstates**(nlayers-m-1)]-c)    
                                else:
                                    newtransbeta3[start+ j*nstates**(nlayers-m-1)] = newtransbeta3[start+ j*nstates**(nlayers-m-1)]+ np.exp(np.log(transP[m,j,i])+ newlogbeta2[start+ i*nstates**(nlayers-m-1)]-c)      

                            else:
                                if transP[m,j,i]==0:
                                    newtransbeta3[start+ j*nstates**(nlayers-m-1)] = newtransbeta3[start+ j*nstates**(nlayers-m-1)]+ np.exp(np.log(transP[m,j,i]+epsilon)+ newlogbeta2[start+ i*nstates**(nlayers-m-1)]+c)       
                                else:
                                    newtransbeta3[start+ j*nstates**(nlayers-m-1)] = newtransbeta3[start+ j*nstates**(nlayers-m-1)]+ np.exp(np.log(transP[m,j,i])+ newlogbeta2[start+ i*nstates**(nlayers-m-1)]+c)                                        

            if c<=0:        
                newlogbeta2 = np.log((newtransbeta3))+ np.exp(c)
            else:
                newlogbeta2 = np.log((newtransbeta3))- np.exp(c)

        newlogbeta1 = newlogbeta2
        logbeta.append(newlogbeta1)

        smooth = np.exp(logbeta[t]+logalpha[T-1-t])
        smoothing.append(smooth/sum(smooth))          
    
    
    return smoothing

#########################################
#########################################

def GJForwardBackwardlog(mu0, transP, W, C, YT):

    filteringDIC = GJforwardlog(mu0, transP, YT, W, C)
    
    smoothing = GJbackwardlog(filteringDIC['logalpha'], transP, YT, W, C)
    
    return dict({'Filtering': filteringDIC['filtering'], 'Smoothing':smoothing})
    
#########################################
#########################################
