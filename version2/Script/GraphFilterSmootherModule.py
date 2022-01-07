import numpy as np
import random
import time
import networkx as nx
from networkx.algorithms import bipartite
import math

# Define an FHMM object
class FHMM:

    # initialize the FHMM
    def __init__(self, factor_graph, states):

        # define the factor graph
        self.factor_graph = factor_graph
        # define the number of states and the state space
        self.n_states = len(states)
        self.states   = states

        self.hidden_variables, self.factors = bipartite.sets(self.factor_graph)
        # define the dimension of the Markov chain and the number of factors
        self.M         = len(self.hidden_variables)
        self.n_factors = len(self.factors)

# Define a GraphFilterSmoother object

class GraphFilterSmoother:

    def __init__(self, factor_graph, states, m, partition, mu_0, transP, c, sigma):

        # define the factor graph
        self.factor_graph = factor_graph
        # define the number of states and the state space
        self.n_states = len(states)
        self.states   = states

        self.hidden_variables, self.factors = bipartite.sets(self.factor_graph)
        # define the dimension of the Markov chain and the number of factors
        self.M         = len(self.hidden_variables)
        self.n_factors = len(self.factors)

        # define the parameters of the graph filter smoother
        self.m = m
        self.partition = partition

        # set the intial conditions
        # for the kernel
        self.mu_0 = mu_0
        self.transP = transP
        #for the emission distribution
        self.c     = c
        self.sigma = sigma

        # define the dictionary of distances according to the parameter m (i.e. all the shortest path of distance 2m+2 of all the nodes)
        self.shortest_path_length = dict(nx.all_pairs_shortest_path_length(self.factor_graph, 2*m+2))

    # find the state k. The total number of states is self.n_states**self.M
    def find_state(self, dimension, k):
        
        state = np.zeros(dimension, dtype=int)
        
        itcomponents = dimension
        j= k
        
        for i in range(0, dimension):
            index = int(np.floor((j)/self.n_states**(itcomponents-1)))
            state[i] = self.states[index]
            j = j - index*(self.n_states**(itcomponents-1))
            itcomponents = itcomponents -1
        
        return state

    def find_state_index(self, dimension, k):
        
        state = np.zeros(dimension, dtype=int)
        
        itcomponents = dimension
        j= k
        
        for i in range(0, dimension):
            index = int(np.floor((j)/self.n_states**(itcomponents-1)))
            state[i] = index
            j = j - index*(self.n_states**(itcomponents-1))
            itcomponents = itcomponents -1
        
        return state

    # compute the set of neighbours distant 2m+1
    def N2m1K(self):
        
        N2m1 = dict()

        for K in range(0, len(self.partition)):
            supp = []
            for element in self.partition[str(K)]:
                for neighbour in self.factors.intersection(self.shortest_path_length[element].keys()):
                    supp.append(neighbour)
        
            N2m1[str(K)] = np.sort(np.array(supp))

        return N2m1

    # compute the set of neighbours distant 2m+2
    def N2m2K(self):
            
        N2m2 = dict()

        for K in range(0,len(self.partition)):
            supp = []
            for element in self.partition[str(K)]:
                for neighbour in self.hidden_variables.intersection(self.shortest_path_length[element].keys()):
                        supp.append(neighbour)

            N2m2[str(K)] = np.sort(np.array(supp))
                      
        return N2m2

    # create the subkernel matrix
    def subsetkernel(self, indexset, transP ):
        
        # create the index set on which we want to joint the subkernel matrices
        newindex = list()
        for i in indexset:
            newindex.append(i)
            
        newindex.sort()
        indexset = newindex
                
        # create the new kernel matrix as a join of the previous matrices
        currentP = transP[indexset[0]]
        
        for index in range(1, len(indexset)):
            
            for i in range(0, np.size(currentP,0)):
                # print(indexset[index])
                supportmatrixrow = currentP[i,0]*transP[indexset[index]]
                
                for j in range(1, np.size(currentP,1)):
                    
                    supportmatrixrow = np.hstack( (supportmatrixrow, (currentP[i,j]*transP[indexset[index]])) )
                    
                if i==0:
                    supportmatrixrowprev = supportmatrixrow
                    
                else:
                    supportmatrixrowprev = np.vstack( (supportmatrixrowprev , supportmatrixrow) )

            currentP = supportmatrixrowprev
                    
        return currentP

    # create all the kernel associated to the set N2m2K 
    def P2m2K(self, N2m2):
        
        P2m2 = dict()
        for K in range(0, len(self.partition)):
            P2m2[str(K)] = self.subsetkernel(N2m2[str(K)], self.transP)  
                
        return P2m2
    
    # create all the initial associated to the set N2m2K 
    def mu_02m2K(self, N2m2):
        
        mu_02m2 = dict()
        for K in range(0, len(self.partition)):
            mu_02m2[str(K)] = self.subsetkernel(N2m2[str(K)], self.mu_0)  
                
        return mu_02m2

    # define the neighbours of level 1 of all the factors
    def N1F(self):

        N1F = dict()

        for factor in self.factors:

            N1F[factor] = []
            for neighbour in self.shortest_path_length[factor].keys():

                if self.shortest_path_length[factor][neighbour] == 1:
                    N1F[factor].append(neighbour)

            N1F[factor] = np.sort(np.array(N1F[factor]))

        return N1F

    # create the the associated mu to the set N2m2K (from a mu that factorizes as in partition create a distribution per each element of N2m2)
    def mu2mK(self, mu, N2m2):

        mu2m2 = dict()
        for K in range(0, len(self.partition)):
            
            currentmu = np.array([float(1)])
            
            if len(N2m2[str(K)])>1:
                for tildeK in range(0, len(self.partition)):

                    if self.partition[str(tildeK)] in N2m2[str(K)]:
                        support2= currentmu[0]* mu[str(tildeK)]
                        for i in range(1, np.size(currentmu)):
                            support1 = currentmu[i]* mu[str(tildeK)]
                            support2 = np.hstack((support2, support1))                    

                        currentmu = support2
            else:
                currentmu = mu[str(K)]

            mu2m2[str(K)] = currentmu
            
        return mu2m2

    # define the emission distribution of a factor
    def emission(self, y, mean, variance):

        return (1/(np.sqrt(2*np.pi*variance)))*np.exp(-(y-mean)**2/(2*variance))

    # define the approximated filtering recursion (graph filter)
    def filtering(self, YT):

        # define the time step
        self.T = len(YT[random.sample(YT.keys(),1)[0]])+1
#         self.T = np.size(YT, 0)

        # define the neighbours
        N2m1  = self.N2m1K()
        N2m2  = self.N2m2K()
        N1F   = self.N1F()

        # define the kernel and current distribution on the neighbours
        P2m2  = self.P2m2K(N2m2)
        
        pifiltering = {}
        
        mu_0_on_K = self.mu_02m2K(self.partition)

        pifiltering["0"] = mu_0_on_K

        for t in range(1, self.T):
#             print(t)
            mu2m2 = self.mu2mK(pifiltering[str(t-1)], self.N2m2K())

            pifiltering[str(t)] = {}

            for K in range(0, len(self.partition)):
                PmuK = np.dot(mu2m2[str(K)] , P2m2[str(K)])
                tildeCPmuK = np.array(PmuK)

                normalizingconst = 0
                for state_number in range(len(PmuK)):

                    state = self.find_state(len(N2m2[str(K)]), state_number)

                    for f in N2m1[str(K)]:

                        mean = 0
                        for x in N1F[f]: 
                            mean += state[x == N2m2[str(K)]]
                        
                        tildeCPmuK[state_number] = tildeCPmuK[state_number]*self.emission(YT[f][t-1], self.c[f]*mean, self.sigma[f])    
                        
#                         tildeCPmuK[state_number] = tildeCPmuK[state_number]*self.emission(YT[t, int(f[1])], self.c[int(f[1])]*mean, self.sigma[int(f[1])])

                    normalizingconst = normalizingconst + tildeCPmuK[state_number]

                tildeCPmuK = tildeCPmuK/normalizingconst

                # marginalize out the useless components
                pifiltering[str(t)][str(K)] = np.zeros(self.n_states**len(self.partition[str(K)]))
                
                if len(self.partition)!=1:
                    numN2m2K = np.zeros(len(N2m2[str(K)]), dtype=int)

                    count = 0
                    for i in N2m2[str(K)]:
                        numN2m2K[count] = int(i[1:])
                        count= count+1

                    numN2m2K= np.sort(numN2m2K)

                    size = len(self.partition[str(K)])
                    pifiltering[str(t)][str(K)] = np.zeros(self.n_states**size)

                    for j in range(0, self.n_states**len(numN2m2K)):
                        current = (self.find_state_index( len(numN2m2K), j))

                        position=0
                        for indexwhich in range(0,size):
                            position = position+ current[numN2m2K==int(self.partition[str(K)][indexwhich][1:])]*(self.n_states**(size-indexwhich-1))
#                         print(position)
#                         print(str(K))
#                         print(str(t))
#                         print(j)
                        pifiltering[str(t)][str(K)][position]= pifiltering[str(t)][str(K)][position]+ tildeCPmuK[j]
                                   
                else:
                    pifiltering[str(t)][str(K)]= tildeCPmuK

        return pifiltering

    # define the approximate smoothing algorithm
    def smoothing(self, pifiltering):
        
        pismoother = {}
        pismoother[str(self.T-1)] = pifiltering[str(self.T-1)]
        
        P2m2  = self.P2m2K(self.partition)
        
        for t in range(1, self.T):                
            
            newsmoother = {}
            for K in range(0, len(self.partition)):

                kernel = P2m2[str(K)]
                
                supportkernel = np.zeros((np.size(kernel,0), np.size(kernel,1)))
                
                for i in range(np.size(kernel, axis=0)):
                    if sum( (pifiltering[str(self.T-t-1)][str(K)])*(kernel[:,i]) )!=0:
                        supportkernel[i,:] = (kernel[:,i]*pifiltering[str(self.T-t-1)][str(K)])/(sum((pifiltering[str(self.T-t-1)][str(K)])*(kernel[:,i])))
                
                probK = np.dot(pismoother[str(self.T-t)][str(K)],  supportkernel)
                
                newsmoother[str(K)] = (probK)
                
            pismoother[str(self.T-1-t)] = (newsmoother)
            
        return pismoother