import numpy as np
import random
import time
import networkx as nx
from networkx.algorithms import bipartite
import math

    
# We are using the trivial partition    
def EMmulti_gaussian( GraphFilterSmoother, YT, iterations ):
    
    nstates       = len(GraphFilterSmoother.states)
    T = len(YT[:,0])
    
    mu0_estimate = []
    for v in range(0, GraphFilterSmoother.M ):
        mu0_estimate.append(GraphFilterSmoother.mu_0[v].copy())
        
    transP_estimate = []
    for v in range(0, GraphFilterSmoother.M ):
        transP_estimate.append( GraphFilterSmoother.transP[v].copy() )

    lamb_estimate = []
    for f in range(0, GraphFilterSmoother.n_factors):
        lamb_estimate.append( GraphFilterSmoother.lamb[f] )
    
    mu_0_list = list()
    mu_0_list.append(mu0_estimate)
    
    transP_list = list()
    transP_list.append(transP_estimate)
    
    lamb_list = list()
    lamb_list.append(lamb_estimate)

    for it in range(1, iterations):  
        print(it)
        
        filtering = GraphFilterSmoother.filtering(YT)
        smoothing = GraphFilterSmoother.smoothing(filtering)
        
        ##############################################################################
        ##############################################################################
        # initial distr
        ##############################################################################        

        # trivial partition = all the hidden units
        
        # all the hidden units have different initial distributions
#         for v in GraphFilterSmoother.hidden_variables:
#                 mu0_estimate[v] = smoothing["0"][v[1:]]
        
        # all the hidden units have same initial distributions        
        mu0_estimate = []
        for v in range(0, GraphFilterSmoother.M ):
            mu0_estimate.append( smoothing[0][v].copy() )
        
        mu_0_list.append(mu0_estimate)
                
        ##############################################################################
        ##############################################################################
        # Kernel
        ##############################################################################
        
        transP_estimate = []
        
        for v in range(0, GraphFilterSmoother.M ):
            
            P_num_estimate = np.zeros((nstates, nstates), dtype = float)
            P_den_estimate = np.zeros((nstates, 1), dtype = float)
            
            for t in range(1, T):

                numjoint = transP_list[it-1][v]*np.array(filtering[t-1][v]).reshape(-1,1)*np.array(smoothing[t][v]).reshape(-1,1).T
                denjoint = np.dot( np.array(filtering[t-1][v]).reshape(-1,1).T, transP_list[it-1][v]).T
                denjoint[denjoint==0] = 1
                    
                jointpit = numjoint/denjoint.T
#                 print(v)
#                 print(filtering[str(t-1)][str(int(v[1:]))].reshape(-1,1))

                P_num_estimate = P_num_estimate + jointpit
                P_den_estimate = P_den_estimate + np.array(smoothing[t-1][v]).reshape(-1,1)
                        
            P_estimate = P_num_estimate/P_den_estimate

            transP_estimate.append( P_estimate.copy() )
            
        transP_list.append(transP_estimate)


        ##############################################################################
        ##############################################################################
        # Lambda
        ##############################################################################
        
        lamb_estimate = []
        
        for f in range(0, GraphFilterSmoother.n_factors):
            
            lamb_num = np.sum(YT[:, f])
            lamb_den = 0
                
            for t in range(1, T):
                                  
                for i in range(nstates**len(GraphFilterSmoother.N1F[f])):

                    current_state = GraphFilterSmoother.find_state( len(GraphFilterSmoother.N1F[f]), i)
                    current_state_index = GraphFilterSmoother.find_state_index( len(GraphFilterSmoother.N1F[f]), i)

                    weight_sum_current_state = np.sum(current_state, dtype = float)

                    index = 0
                    for v in GraphFilterSmoother.N1F[f]:
                        weight_sum_current_state = weight_sum_current_state*smoothing[t][v][current_state_index[index]]
                        index = index + 1
                        
                    lamb_den = lamb_den + weight_sum_current_state
                    
            lamb_single_estimate = lamb_num/lamb_den#
            
            lamb_estimate.append(lamb_single_estimate)
            
        lamb_list.append(lamb_estimate)


        
        GraphFilterSmoother.mu_0   = mu_0_list[it]
        GraphFilterSmoother.transP = transP_list[it]
        GraphFilterSmoother.lamb   = lamb_list[it]

    return dict({'mu_0_list': mu_0_list, 'transP_list': transP_list, 'lamb_list': lamb_list})    