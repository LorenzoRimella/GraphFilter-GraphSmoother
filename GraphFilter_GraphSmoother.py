import numpy as np
import random
import time
import networkx as nx
from networkx.algorithms import bipartite
import math

#########################################
#########################################

def findvalue(nstates, ncomponents, k):
    
    x = np.zeros(ncomponents, dtype=int)
    
    itcomponents = ncomponents
    j= k
    
    for i in range(0, ncomponents):
        x[i] = int(np.floor((j)/nstates**(itcomponents-1)))
        j = j - x[i]*(nstates**(itcomponents-1))
        itcomponents = itcomponents -1
    
    return x

#########################################
#########################################

def simArray(mu0):
    
    csmu0 = np.cumsum(mu0)
    prob = random.random()
    
    return(int(sum(csmu0 < prob)))

#########################################
#########################################

def transMat(transP, xprev):
    
    return(int(simArray(np.asarray(transP[xprev,:]))))

#########################################
#########################################

def supportVec(nstates, position):
    output           = np.zeros(nstates, dtype=float)
    output[position] = 1
    return(np.transpose(output))

#########################################
#########################################

def FHMM(T, nstates, nlayers, dimObs, mu0, transP, W, C):
    
    hiddenstates = np.zeros((T, nlayers), dtype=int)
    observations = np.zeros((T, dimObs))
    
    for i in range(0,nlayers):
        hiddenstates[0, i] = simArray(mu0[i,:]) #remark that mu0 is an array of vector of probabilitites
    
    time = 1
    while(time < T):
        
        mean = np.zeros(dimObs)
        for i in range(0,nlayers):
            transMatrix = np.asmatrix(transP[i, :, :])
            hiddenstates[time, i] = transMat(transMatrix, hiddenstates[time-1, i])
            mean = mean + np.asarray(W[i,:,hiddenstates[time, i]])
        
        observations[time,:] = np.random.multivariate_normal(mean, C)
        
        time = time+1
        
    return {'hidden':hiddenstates,'observations':observations}

#########################################
#########################################

def subsetkernel(indexset, transP, transformIndex=True):
    
    if transformIndex == True:
        newindex = list()
        for i in indexset:
            newindex.append(int(i[1:]))
            
        newindex.sort()
        indexset = newindex
        # print(indexset)
            
    
    currentP = transP[indexset[0], :, :]
    
    for index in range(1, len(indexset)):
        
        for i in range(0, np.size(currentP,0)):
            
            supportmatrixrow = currentP[i,0]*transP[indexset[index], :, :]
            
            for j in range(1, np.size(currentP,1)):
                supportmatrixrow = np.hstack( (supportmatrixrow, (currentP[i,j]*transP[indexset[index], :, :])) )
                
            if i==0:
                supportmatrixrowprev = supportmatrixrow
                
            else:
                supportmatrixrowprev = np.vstack( (supportmatrixrowprev , supportmatrixrow) )

        currentP = supportmatrixrowprev
                  
    return currentP

#########################################
#########################################

def nfactor(C):
    
    factors = {}
    setindex = range(0, np.size(C,1))
    
    ifa = 0
    iC = 0
    while(len(setindex)>0):
        # create a dictionary with key given by the factor number
        factors['F'+str(ifa)] = np.extract(C[setindex[0],:]!=0, range(0, np.size(C,1)) )
        
        setindex= np.setdiff1d(setindex, factors['F'+str(ifa)])
        ifa = ifa+1
    
    return factors

#########################################
#########################################

def observVShid(W):
    W= abs(W)
    obshid = {}
    
    for i in range(0, np.size(W,0)):
        # sum by row W and find the one that has some sor of contribution from the hidden state
        obshid['H'+str(i)] = np.extract(W[i,:,:].sum(axis=1)>0, range(0, np.size(W,1)) )
    
    return obshid

#########################################
#########################################

def bipartitegraph(C, W, factor= True):

    nlayers = np.size(W,0)
    
    nobserv = np.size(C,0)
    
    HIDDEN = observVShid(W)
    
    FACTOR = nfactor(C)
    
    # Create the graph
    G = nx.Graph()
    
    # Add nodes
    G.add_nodes_from(set(HIDDEN.keys()), bipartite=0) # Add the node attribute "bipartite"
    G.add_nodes_from(set(FACTOR.keys()), bipartite=1)
    
    # Add edges
    for node1 in set(HIDDEN.keys()):
        for node2 in set(FACTOR.keys()):
            
            if len(set(HIDDEN[node1]) & set(FACTOR[node2]))>0:
                G.add_edges_from([(node1,node2)])
                
    if factor== False:
        return G
    
    else:
        return dict({'Graph': G, 'factor':FACTOR})
    
#########################################
#########################################

# Construct N^2m+1(K)
def N2m1K(G, partition, m, spl, keys=False):
    
    if keys==True:
        FACTOR = set(n for n,d in G.nodes(data=True) if d['bipartite']==1)
        N2m1 = {}
    
        for K in partition.keys():
            supp = set()
            for element in partition[K]:
                for neighbour in FACTOR.intersection(spl[element].keys()):
                    if(spl[neighbour][element] <= 2*m+1 ):
                        supp.add(neighbour)
        
            N2m1[K]= supp
            
    #########################
    else:
        FACTOR = set(n for n,d in G.nodes(data=True) if d['bipartite']==1)
        N2m1 = list()

        for K in range(0,len(partition)):
            supp = set()
            for element in partition[K]:
                for neighbour in FACTOR.intersection(spl[element].keys()):
                    if(spl[neighbour][element] <= 2*m+1 ):
                        supp.add(neighbour)
        
            N2m1.append(supp)
        
        
    return N2m1

#########################################
#########################################

# Construct N^2m+2(K)
def N2m2K(G, partition, m, spl, keys=False):
    
    if keys==True:
        HIDDEN = set(n for n,d in G.nodes(data=True) if d['bipartite']==0)
        N2m2 = {}
    
        for K in partition.keys():
            supp = set()
            for element in partition[K]:
                for neighbour in HIDDEN.intersection(spl[element].keys()):
                    if(spl[neighbour][element] <= 2*m+2 ):
                        supp.add(neighbour)
        
            N2m2[K]= supp
            
    #########################
    else:
        HIDDEN = set(n for n,d in G.nodes(data=True) if d['bipartite']==0)
        N2m2 = list()

        for K in range(0,len(partition)):
            supp = set()
            for element in partition[K]:
                for neighbour in HIDDEN.intersection(spl[element].keys()):
                    if(spl[neighbour][element] <= 2*m+2 ):
                        supp.add(neighbour)
        
            N2m2.append(supp)
        
        
    return N2m2

#########################################
#########################################

# Kernel on the graph structure

def P2m2K(N2m2, transP):

    if type(N2m2)!= list:
        P2m2 = {}
        for K in N2m2.keys():
            P2m2[K] = subsetkernel(N2m2[K], transP, transformIndex= False)
            
    else:
        P2m2 = list()
        for K in range(0, len(N2m2)):
            P2m2.append(subsetkernel(N2m2[K], transP, transformIndex= True) )       
            
    return P2m2

#########################################
#########################################

def mu2mK(N2m2inK, mu, partition):
    
    recovery1 = set()
    recovery2 = list()
    
    i=0
    while not(N2m2inK.issubset(recovery1)):
        if len(set(partition[i]).intersection(N2m2inK))!=0:
            recovery1= recovery1.union(set(partition[i]))
            recovery2.append(i)
        i=i+1

    recoveryindex = recovery2
    
    del recovery1
    del recovery2
    
    currentmu = np.array([float(1)])
    
    if len(recoveryindex)>1:
        for index in recoveryindex:
        
            support2= currentmu[0]* mu[index]
            for i in range(1, np.size(currentmu)):
                support1 = currentmu[i]* mu[index]
                support2 = np.hstack((support2, support1))
            
            currentmu = support2
    else:
        currentmu = mu[recoveryindex[0]]
        
    return currentmu

#########################################
#########################################

def Gtx(obs, yobs, C, W, x, HID):
    
    mu = np.zeros(len(obs))
    
    for i in range(0, len(obs)):
        
        mu[i] = 0
        help1=0
        for j in HID:
            
            mu[i] = mu[i] + (W[j,obs[i],x[help1]])
            help1= help1+1
            
        #print('mean: ', mu)
         
    
    return float(np.exp(-0.5*np.dot(np.dot(((yobs-mu).T),(np.asmatrix(C).I)[obs,:][:,obs]),(yobs-mu))))

#########################################
#########################################

def GtxCompleate(obs, yobs, C, W, x, HID):
    
    mu = np.zeros(len(obs))
    
    for i in range(0, len(obs)):
        
        mu[i] = 0
        help1=0
        for j in HID:
            
            mu[i] = mu[i] + (W[j,obs[i],x[help1]])
            help1= help1+1
            
        #print('mean: ', mu)
         
    
    return float(((2*math.pi*np.linalg.det(C))**(-1/2))*np.exp(-0.5*np.dot(np.dot(((yobs-mu).T),(np.asmatrix(C).I)[obs,:][:,obs]),(yobs-mu))))

#########################################
#########################################

def GraphFilter(mu0, transP, partition, W, C, m, YT):
    
    
    nlayers = np.size(W, 0)
    nstates = np.size(W, 2)
    nobservations = np.size(YT, 1)
    T = np.size(YT, 0)

    # Learn the kernel on the graph structure
    Gdict= bipartitegraph(np.asmatrix(C).I, abs(W))
    G = Gdict['Graph']
    spl = dict(nx.all_pairs_shortest_path_length(G))
    HIDDEN = observVShid(W)
    FACTOR = nfactor(C)
    
    N2m2 = N2m2K(G, partition, m, spl, keys=False)
    N2m1 = N2m1K(G, partition, m, spl, keys=False)
    P2m2 = P2m2K(N2m2, transP)
    
    pi = list()
    
    # Initial condition
    trivialpart = list()
    for i in list(HIDDEN.keys()):
        trivialpart.append([i])
    
    newmu0 = list()
    for K in partition:
        newmu0.append(mu2mK(set(K), mu0, trivialpart))
        
    pi.append(newmu0)
    
    for t in range(1,T):
      
        newpi = list()
        for K in range(0, len(partition)):
            
            piprimet = mu2mK(N2m2[K], pi[t-1], partition)
            piprimet = np.dot(piprimet , P2m2[K])
            
            normalizingconst = 0
            
            
            N2m2array = np.zeros(len(N2m2[K]), dtype=int)
            counter=0
            for name in N2m2[K]:
                N2m2array[counter]= int(name[1:]) 
                counter= counter+1
            N2m2array= np.sort(N2m2array)
            
            
            intersection = N2m2[K].intersection(set(partition[K]))
            index = np.zeros(len(intersection), dtype=int)
            counter=0
            for i in intersection:
                index[counter] = int(i[1:])
                counter= counter+1 
            
            index = np.sort(index)
            
            for k in range(0, len(piprimet)):
                    
                x= findvalue(nstates, len(N2m2[K]), k)
               
                for f in N2m1[K]:
                    piprimet[k] =piprimet[k]*Gtx(FACTOR[f], YT[t][FACTOR[f]], C, W, x, N2m2array)
                    
                normalizingconst = normalizingconst + piprimet[k] 
            
            piprimet = piprimet/normalizingconst
            
            
            margpiprimet = np.zeros(nstates**len(partition[K]))
            
            if len(partition)!=1:
                numN2m2K = np.zeros(len(N2m2[K]), dtype=int)

                count = 0
                for i in N2m2[K]:
                    numN2m2K[count] = int(i[1:])
                    count= count+1

                numN2m2K= np.sort(numN2m2K)

                size = len(partition[K])
                margpiprimet = np.zeros(nstates**size)

                for j in range(0, nstates**len(numN2m2K)):
                    current = (findvalue(nstates, len(numN2m2K), j))

                    position=0
                    for indexwhich in range(0,size):
                        position = position+ current[numN2m2K==int(partition[K][indexwhich][1:])]*(nstates**(size-indexwhich-1))

                    margpiprimet[position]= margpiprimet[position]+ piprimet[j]
                    
            
            else:
                margpiprimet= piprimet
                
            newpi.append(margpiprimet)  
            
        pi.append(newpi)
        
    return pi


#########################################
#########################################

def GraphSmoother(pifiltering, transP, partition, T):

    pismoother = list()
    pismoother.append(pifiltering[T-1])
    
    for t in range(1,T):                

        newsmoother = list()
        for K in partition:
            
            kernel = subsetkernel(K, transP, transformIndex=True)
            supportkernel = np.zeros((np.size(kernel,0), np.size(kernel,1)))
            
            for i in range(np.size(kernel, axis=0)):
                if sum( (pifiltering[T-t-1][partition.index(K)])*(kernel[:,i]) )!=0:
                    supportkernel[i,:] = (kernel[:,i]*pifiltering[T-t-1][partition.index(K)])/(sum((pifiltering[T-t-1][partition.index(K)])*(kernel[:,i])))
            
            probK = np.dot(pismoother[t-1][partition.index(K)],  supportkernel)
            
            newsmoother.append(probK)
            
        pismoother.append(newsmoother)
        
    return pismoother


#########################################
#########################################

def GraphForwardBackward(mu0, transP, partition, W, C, m, YT):

    filtering = GraphFilter(mu0, transP, partition, W, C, m, YT)
    
    smoothing = GraphSmoother(filtering, transP, partition, np.size(YT,0))
    
    return dict({'Filtering': filtering, 'Smoothing':smoothing})

#########################################
#########################################
