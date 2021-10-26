import numpy as np

import GraphFilter_GraphSmoother

import pickle

def EMtoy(nlayers, YT, intial0, P0, c0, sigma0, partition, m, iterations):
    
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

    for it in range(1, iterations):    
        DICTO = GraphFilter_GraphSmoother.GraphForwardBackward(mu00, transP0, partition, W0, C0, m, YT)

        smoothing = DICTO['Smoothing']
        filtering = DICTO['Filtering']

        # initial distr
        mu0prov = np.zeros((nstates), dtype = float)
        for i in range(0, nstates):
            for v in range(0,nlayers):
                mu0prov[i] = mu0prov[i] + smoothing[T-1][v][i]
        
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
                jointpit = np.zeros((nstates, nstates), dtype = float)
                for j in range(0, nstates):
                    denjoint = sum(P[it-1][:,j]*filtering[t-1][v])
                    for i in range(0, nstates):
                        if denjoint==0 and P[it-1][i,j]*filtering[t-1][v][i]*smoothing[T-t-1][v][j]==0:
                            jointpit[i, j] = 0
                        else:
                            jointpit[i, j] = P[it-1][i,j]*filtering[t-1][v][i]*smoothing[T-t-1][v][j]/denjoint

                for i in range(nstates):
                    for j in range(nstates):
                        Pprovnum[i,j] = jointpit[i,j] + Pprovnum[i,j]
                        Pprovden[i,j] = smoothing[T-t][v][i] + Pprovden[i,j]
                        
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
                    num = num + YT[t][f]*sum(state+1)*smoothing[T-t-1][f][state[0]]*smoothing[T-t-1][f+1][state[1]]
                    den = den + ((sum(state+1))**2)*smoothing[T-t-1][f][state[0]]*smoothing[T-t-1][f+1][state[1]]
                    
        cprov = num/den
        c.append(cprov)

        W0 = cprov*W
        
        
        sigmaprov = 0
        for t in range(1, T):
            for f in range(0, nobservations):    
                for i in range(nstates**2):
                    state = GraphFilter_GraphSmoother.findvalue(nstates, 2, i)
                    sigmaprov = sigmaprov + ((YT[t][f] - cprov*sum(state+1))**2)*smoothing[T-t-1][f][state[0]]*smoothing[T-t-1][f+1][state[1]]
                    
        sigmaprov = sigmaprov/((T-1)*nobservations)
        
        sigma.append(sigmaprov)
 
        C0 = sigmaprov*np.diag(np.ones(nobservations))
    
    
    
    return dict({'mu0': mu0, 'P': P, 'c': c, 'sigma': sigma})






    
def EMlondon(nlayers, YT, intial0, P0, c0, sigma0, partition, m, iterations, statevar):
    
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
    
    W = np.array([[statevar],
                  [statevar]], dtype=float)
    

    for ite in range(2, nlayers):
        newW = np.zeros((ite+1,ite,nstates), dtype=float)
        newW[0:ite, 0:ite-1, :] = W
        newW[ite-1, ite-1, :] = W[0,0,:]
        newW[ite, ite-1, :]   = W[0,0,:]

        W = newW  
        
    W0 = c0*W

    for it in range(1, iterations):  
        print(it)
        DICTO = GraphFilter_GraphSmoother.GraphForwardBackward(mu00, transP0, partition, W0, C0, m, YT)

        smoothing = DICTO['Smoothing']
        filtering = DICTO['Filtering']

        # initial distr
        mu0prov = np.zeros((nstates), dtype = float)
        for i in range(0, nstates):
            for v in range(0,nlayers):
                mu0prov[i] = mu0prov[i] + smoothing[T-1][v][i]
        
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
                jointpit = np.zeros((nstates, nstates), dtype = float)
                for j in range(0, nstates):
                    denjoint = sum(P[it-1][:,j]*filtering[t-1][v])
                    for i in range(0, nstates):
                        if denjoint==0 and P[it-1][i,j]*filtering[t-1][v][i]*smoothing[T-t-1][v][j]==0:
                            jointpit[i, j] = 0
                        else:
                            jointpit[i, j] = P[it-1][i,j]*filtering[t-1][v][i]*smoothing[T-t-1][v][j]/denjoint

                for i in range(nstates):
                    for j in range(nstates):
                        Pprovnum[i,j] = jointpit[i,j] + Pprovnum[i,j]
                        Pprovden[i,j] = smoothing[T-t][v][i] + Pprovden[i,j]
                        
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
                    state1 = np.array([statevar[state[0]], statevar[state[1]]], dtype= float)

                    num = num + YT[t][f]*sum(state1)*smoothing[T-t-1][f][state[0]]*smoothing[T-t-1][f+1][state[1]]
                    den = den + ((sum(state1))**2)*smoothing[T-t-1][f][state[0]]*smoothing[T-t-1][f+1][state[1]]
                    
        cprov = num/den
        c.append(cprov)

        W0 = cprov*W
 
        sigmaprov = 0
        
        for t in range(1, T):
            for f in range(0, nobservations):    
                for i in range(nstates**2):
                    state = GraphFilter_GraphSmoother.findvalue(nstates, 2, i)
                    state1 = np.array([statevar[state[0]], statevar[state[1]]], dtype= float)

                    sigmaprov = sigmaprov + ((YT[t][f] - cprov*sum(state1))**2)*smoothing[T-t-1][f][state[0]]*smoothing[T-t-1][f+1][state[1]]  
                    
        sigmaprov = sigmaprov/((T-1)*nobservations)
        
        sigma.append(sigmaprov)
 
        C0 = sigmaprov*np.diag(np.ones(nobservations))
    
    return dict({'mu0': mu0, 'P': P, 'c': c, 'sigma': sigma})  