import numpy as np
import random
import time
import math
import GraphFilter_GraphSmoother
import GharamaniJordan
import GJsubroutine


def softmax(A):
    return np.exp(A)/sum(np.exp(A))

def variationalBayes(nlayers, YT, intial0, P0, c0, sigma0, variational, iterations):
    
    nstates = np.size(P0, 1)
    nobservations = np.size(YT, 1)
    T = np.size(YT, 0)
    
    mu0 = list()
    mu0.append(intial0)
    
    P = list()
    P.append(P0)
    
    c = list()
    c.append(c0)
    
    sigma = list()
    sigma.append(sigma0)
    
    mu00 = np.zeros((nlayers,nstates), dtype = float)
    for i in range(0, nlayers):
        mu00[i,:] = intial0
        
    transP0 = np.zeros((nlayers,nstates,nstates), dtype = float)
    for i in range(0, nlayers):
        transP0[i,:,:] = P0
    
    C0 = sigma0*np.diag(np.ones(nobservations))
    
    W = np.array([[np.array(range(nstates), dtype = float)+1],
                  [np.array(range(nstates), dtype = float)+1]], dtype=float)
    
    for ite in range(2, nlayers):
        # matrix W
        newW = np.zeros((ite+1,ite,nstates), dtype=float)
        newW[0:ite, 0:ite-1, :] = W
        newW[ite-1, ite-1, :] = W[0, 0, :]
        newW[ite, ite-1, :]   = W[0, 0, :]

        W = newW
        
    W0 = c0*W
    
    if variational['type'] == 'one':
        oldtheta = variational['initial']
        newtheta = variational['initial']
        
    else:
        oldh    = variational['initial']
        newh    = variational['initial']
        Pprov   = P0
        mu0prov = intial0
        
    for it in range(1, iterations):    
        if variational['type']== 'one':
            for itprime in range(0, variational['iteration']):
                
                for layer1 in range(0,nlayers):
                    YTtildem = YT[0,:]
                    for layer2 in range(0,nlayers):
                        if layer2 != layer1:
                            YTtildem = YTtildem - np.dot(W0[layer2,:,:], oldtheta[0, layer2, :])
                        
                    exponent = np.dot(np.dot((W0[layer1,:,:]).T, np.diag(1/np.diag(C0))), YTtildem)
                    exponent = exponent - 0.5*(np.diag(np.dot(np.dot((W0[layer1,:,:]).T, np.diag(1/np.diag(C0))), W0[layer1,:,:])))
                    exponent = exponent + np.log(mu00[0,:])
                    exponent = exponent + np.dot(np.log((transP0[layer1,:,:]).T), oldtheta[1, layer1, :])
                    
                    newtheta[0,layer1,:] = softmax(exponent)
                
                for tprime in range(1,T-1):
                    for layer1 in range(0,nlayers):
                        YTtildem = YT[tprime,:]
                        for layer2 in range(0,nlayers):
                            if layer2 != layer1:
                                YTtildem = YTtildem - np.dot(W0[layer2,:,:], oldtheta[tprime, layer2, :])

                        exponent = np.dot(np.dot((W0[layer1,:,:]).T, np.diag(1/np.diag(C0))), YTtildem)
                        exponent = exponent - 0.5*(np.diag(np.dot(np.dot((W0[layer1,:,:]).T, np.diag(1/np.diag(C0))), W0[layer1,:,:])))
                        exponent = exponent + np.dot(np.log(transP0[layer1,:,:]), oldtheta[tprime-1, layer1, :])
                        exponent = exponent + np.dot(np.log((transP0[layer1,:,:]).T), oldtheta[tprime+1, layer1, :])

                        newtheta[tprime,layer1,:] = softmax(exponent)  
                        
                tprime= T-1       
                for layer1 in range(0,nlayers):
                    YTtildem = YT[tprime,:]
                    for layer2 in range(0,nlayers):
                        if layer2 != layer1:
                            YTtildem = YTtildem - np.dot(W0[layer2,:,:], oldtheta[tprime, layer2, :])

                    exponent = np.dot(np.dot((W0[layer1,:,:]).T, np.diag(1/np.diag(C0))), YTtildem)
                    exponent = exponent - 0.5*(np.diag(np.dot(np.dot((W0[layer1,:,:]).T, np.diag(1/np.diag(C0))), W0[layer1,:,:])))
                    exponent = exponent + np.dot(np.log(transP0[layer1,:,:]), oldtheta[tprime-1, layer1, :])

                    newtheta[tprime,layer1,:] = softmax(exponent)   
                    
                oldtheta = newtheta
           
            smoothing = newtheta
            
            # initial distr
            mu0prov = np.zeros((nstates), dtype = float)
            for i in range(0, nstates):
                for v in range(0,nlayers):
                    mu0prov[i] = mu0prov[i] + smoothing[T-1,v,i]

            mu0prov = mu0prov/nlayers

            mu0.append(mu0prov)

            mu00 = np.zeros((nlayers,nstates), dtype = float)
            for i in range(0, nlayers):
                mu00[i,:] = mu0prov

            # Kernel
            Pprovnum = np.zeros((nstates,nstates), dtype = float)
            Pprovden = np.zeros((nstates,nstates), dtype = float)

            for t in range(1, T):
                for v in range(0, nlayers):
                    jointpit = np.dot(np.asmatrix(smoothing[T-t,v,:]).T,np.asmatrix(smoothing[T-t-1,v,:]))

                    for i in range(nstates):
                        for j in range(nstates):
                            Pprovnum[i,j] = jointpit[i,j] + Pprovnum[i,j]
                            Pprovden[i,j] = smoothing[T-t,v,i] + Pprovden[i,j]

            Pprov = Pprovnum/Pprovden

            P.append(Pprov)

            transP0 = np.zeros((nlayers,nstates,nstates),dtype = float)
            for i in range(0, nlayers):
                transP0[i,:,:] = Pprov

            # c and sigma
            num = 0
            den = 0

            for t in range(1, T):
                for f in range(0, nobservations):    
                    for i in range(nstates**2):
                        state = GraphFilter_GraphSmoother.findvalue(nstates, 2, i)
                        num = num + YT[t][f]*sum(state+1)*smoothing[T-t-1,f,state[0]]*smoothing[T-t-1,f+1,state[1]]
                        den = den + ((sum(state+1))**2)*smoothing[T-t-1,f,state[0]]*smoothing[T-t-1,f+1,state[1]]

            cprov = num/den
            c.append(cprov)

            W0 = cprov*W


            sigmaprov = 0
            for t in range(1, T):
                for f in range(0, nobservations):    
                    for i in range(nstates**2):
                        state = GraphFilter_GraphSmoother.findvalue(nstates, 2, i)
                        sigmaprov = sigmaprov + ((YT[t][f] - cprov*sum(state+1))**2)*smoothing[T-t-1,f,state[0]]*smoothing[T-t-1,f+1,state[1]]

            sigmaprov = sigmaprov/((T-1)*nobservations)

            sigma.append(sigmaprov)

            C0 = sigmaprov*np.diag(np.ones(nobservations))                
                
            
        else:
            
            
            for itprime in range(0, variational['iteration']):
                                
                forwardbackward = list()
                for m in range(0,nlayers):
                    subh = np.zeros((T,1,nstates), dtype = float)
                    for i in range(0,T):
                        for j in range(0,nstates):
                            subh[i,0,j] = oldh[i,m,j]
                            
                    forwardbackward.append(GJsubroutine.srGJ2(mu0prov, Pprov, subh, T))          
                
                
                for tprime in range(1,T):
                    for layer1 in range(0,nlayers):
                        YTtildem = YT[tprime,:]
                        
                        for layer2 in range(0,nlayers):
                            if layer2 != layer1:
                                YTtildem = YTtildem - np.dot(W0[layer2, :,:], forwardbackward[layer2][tprime])
                            
                        exponent = np.dot(np.dot((W0[layer1,:,:]).T, np.diag(1/np.diag(C0))), YTtildem)
                        exponent = exponent - 0.5*(np.diag(np.dot(np.dot((W0[layer1,:,:]).T, np.diag(1/np.diag(C0))), W0[layer1,:,:])))
                        
                        newh[tprime,layer1,:] = np.exp(exponent) 
                        
                oldh = newh                
                
                
            filtering = np.zeros((T,nlayers,nstates),dtype= float)
            smoothing = np.zeros((T,nlayers,nstates),dtype= float)     
                                                          
            for l in range(0,nlayers):
                filtering[0,l,:]= mu0prov[:]

            for t in range(1,T):
                for l in range(0,nlayers):
                    normt = 0.0
                    filtering[t,l,:]= newh[t,l,:]*np.dot(filtering[t-1,l,:],Pprov)
                    filtering[t,l,:] = filtering[t,l,:]/sum(filtering[t,l,:])
                    
                    
            jointpit = list()             
            smoothing[T-1,:,:] = filtering[T-1,:,:]
            for t in range(1,T):
                jointpitLAYER = list()
                for l in range(0,nlayers):
                    layerjoint = np.zeros((nstates,nstates),dtype=float)
                    for s1 in range(0,nstates):
                        normjoint = 0
                        
                        for s2 in range(0,nstates):
                            layerjoint[s2,s1] = Pprov[s2,s1]*filtering[T-t-1,l,s2]*smoothing[T-t,l,s1]
                            normjoint = normjoint+Pprov[s2,s1]*filtering[T-t-1,l,s2]
                            
                        layerjoint[:,s1]= layerjoint[:,s1]/normjoint
                        
                    smoothing[T-t-1,l,:]= layerjoint.sum(axis=0)   
                    jointpitLAYER.append(layerjoint)
                jointpit.append(jointpitLAYER)   
                        
                            
                    
                    
                
                        
            # initial distr
            mu0prov = np.zeros((nstates), dtype = float)
            for i in range(0, nstates):
                for v in range(0,nlayers):
                    mu0prov[i] = mu0prov[i] + smoothing[0,v,i]

            mu0prov = mu0prov/nlayers

            mu0.append(mu0prov)

            mu00 = np.zeros((nlayers,nstates), dtype = float)
            for i in range(0, nlayers):
                mu00[i,:] = mu0prov

            # Kernel
            Pprovnum = np.zeros((nstates,nstates), dtype = float)
            Pprovden = np.zeros((nstates,nstates), dtype = float)

            for t in range(1, T):
                for v in range(0, nlayers):

                    for i in range(nstates):
                        for j in range(nstates):
                            Pprovnum[i,j] = jointpit[T-t-1][v][j,i] + Pprovnum[i,j]
                            Pprovden[i,j] = smoothing[t-1,v,i] + Pprovden[i,j]

            Pprov = Pprovnum/Pprovden

            P.append(Pprov)

            transP0 = np.zeros((nlayers,nstates,nstates),dtype = float)
            for i in range(0, nlayers):
                transP0[i,:,:] = Pprov

            # c and sigma
            num = 0
            den = 0

            for t in range(1, T):
                for f in range(0, nobservations):    
                    for i in range(nstates**2):
                        state = GraphFilter_GraphSmoother.findvalue(nstates, 2, i)
                        num = num + YT[t][f]*sum(state+1)*smoothing[t,f,state[0]]*smoothing[t,f+1,state[1]]
                        den = den + ((sum(state+1))**2)*smoothing[t,f,state[0]]*smoothing[t,f+1,state[1]]

            cprov = num/den
            c.append(cprov)

            W0 = cprov*W


            sigmaprov = 0
            for t in range(1, T):
                for f in range(0, nobservations):    
                    for i in range(nstates**2):
                        state = GraphFilter_GraphSmoother.findvalue(nstates, 2, i)
                        sigmaprov = sigmaprov + ((YT[t][f] - cprov*sum(state+1))**2)*smoothing[t,f,state[0]]*smoothing[t,f+1,state[1]]

            sigmaprov = sigmaprov/((T-1)*nobservations)

            sigma.append(sigmaprov)

            C0 = sigmaprov*np.diag(np.ones(nobservations))                            
                            
    
    return dict({'mu0': mu0, 'P': P, 'c': c, 'sigma': sigma})