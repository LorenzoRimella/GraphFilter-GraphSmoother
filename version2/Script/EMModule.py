import numpy as np
import random
import time
import networkx as nx
from networkx.algorithms import bipartite
import math
import pickle as pkl
    
# We are using the trivial partition    
def EMmulti_gaussian( GraphFilterSmoother, YT, iterations ):
    
    nstates       = len(GraphFilterSmoother.states)
    nobservations = len(YT.keys())
    T             = len(YT["F00"])

    ############################################################################
    title = "GraphFilterSmoother-VarParameters-101-"
    title = title + "lenStateSpace-"+str(nstates)+".txt"       
    f= open(title,"a")
    f.close()
    ############################################################################
    
    N1F = GraphFilterSmoother.N1F()
    
    mu0_estimate = {}
    for v in GraphFilterSmoother.hidden_variables:
        mu0_estimate[v] = GraphFilterSmoother.mu_0[v].copy()
        
    transP_estimate = {}
    for v in GraphFilterSmoother.hidden_variables:
        transP_estimate[v] = GraphFilterSmoother.transP[v].copy()

    c_estimate = {}
    for f in GraphFilterSmoother.factors:
        c_estimate[f] = GraphFilterSmoother.c[f]

    sigma_estimate = {}
    for f in GraphFilterSmoother.factors:
        sigma_estimate[f] = GraphFilterSmoother.sigma[f]
    
    mu_0_list = list()
    mu_0_list.append(mu0_estimate)
    
    transP_list = list()
    transP_list.append(transP_estimate)
    
    c_list = list()
    c_list.append(c_estimate)
    
    sigma_list = list()
    sigma_list.append(sigma_estimate)

    for it in range(1, iterations):  
        string = ["Iteration: "+ str(it), "\n"]
        f= open(title,"a")
        f.writelines(string)
        f.close()
       
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
        mu0_estimate = {}
        for v in GraphFilterSmoother.hidden_variables:
            mu0_estimate[v] = smoothing["0"][str(int(v[1:]))]
        
        mu_0_list.append(mu0_estimate)
                
        ##############################################################################
        ##############################################################################
        # Kernel
        ##############################################################################
        
        transP_estimate = {}
        
        for v in GraphFilterSmoother.hidden_variables:
            
            P_num_estimate = np.zeros((nstates, nstates), dtype = float)
            P_den_estimate = np.zeros((nstates, 1), dtype = float)
            
            for t in range(1, T):

                numjoint = transP_list[it-1][v]*filtering[str(t-1)][str(int(v[1:]))].reshape(-1,1)*smoothing[str(t)][str(int(v[1:]))].reshape(-1,1).T
                denjoint = np.dot( filtering[str(t-1)][str(int(v[1:]))].reshape(-1,1).T, transP_list[it-1][v]).T
                denjoint[denjoint==0] = 1
                    
                jointpit = numjoint/denjoint.T
#                 print(v)
#                 print(filtering[str(t-1)][str(int(v[1:]))].reshape(-1,1))

                P_num_estimate = P_num_estimate + jointpit
                P_den_estimate = P_den_estimate + smoothing[str(t-1)][str(int(v[1:]))].reshape(-1,1)
                        
            P_estimate = P_num_estimate/P_den_estimate

            transP_estimate[v] = P_estimate
            
        transP_list.append(transP_estimate)

        ##############################################################################
        ##############################################################################
        # c
        ##############################################################################
        
        c_estimate = {}
        for f in GraphFilterSmoother.factors:
            
            c_state_sum_obs_num_t = 0
            c_state_sum_obs_den_t = 0
                
            for t in range(1, T):
                                  
                for i in range(nstates**len(N1F[f])):

                    current_state = GraphFilterSmoother.find_state( len(N1F[f]), i)
                    current_state_index = GraphFilterSmoother.find_state_index( len(N1F[f]), i)

                    c_state_sum_obs_num = YT[f][t]*np.sum(current_state, dtype = float)
                    c_state_sum_obs_den = (np.sum(current_state, dtype = float))**2

                    index = 0
                    for v in N1F[f]:
                        c_state_sum_obs_num = c_state_sum_obs_num*smoothing[str(t)][str(int(v[1:]))][current_state_index[index]]
                        c_state_sum_obs_den = c_state_sum_obs_den*smoothing[str(t)][str(int(v[1:]))][current_state_index[index]]
                        index = index + 1
                        
                    c_state_sum_obs_num_t = c_state_sum_obs_num_t + c_state_sum_obs_num
                    c_state_sum_obs_den_t = c_state_sum_obs_den_t + c_state_sum_obs_den
                    
            c_single_estimate = c_state_sum_obs_num_t/c_state_sum_obs_den_t#
            
            c_estimate[f] = c_single_estimate
            
        c_list.append(c_estimate)

        ##############################################################################
        ##############################################################################
        # sigma
        ##############################################################################
        
        sigma_estimate = {}
        
        for f in GraphFilterSmoother.factors:
        
            sigma_state_sum_obs_num_t = 0
                
            for t in range(1, T):
                                  
                for i in range(nstates**len(N1F[f])):

                    current_state = GraphFilterSmoother.find_state( len(N1F[f]), i)
                    current_state_index = GraphFilterSmoother.find_state_index( len(N1F[f]), i)

                    sigma_state_sum_obs_num = ( YT[f][t] - c_single_estimate*np.sum(current_state, dtype = float) )**2

                    index = 0
                    for v in N1F[f]:
                        sigma_state_sum_obs_num = sigma_state_sum_obs_num*smoothing[str(t)][str(int(v[1:]))][current_state_index[index]]
                        index = index + 1
                        
                    sigma_state_sum_obs_num_t = sigma_state_sum_obs_num_t + sigma_state_sum_obs_num
                    
            sigma_single_estimate = sigma_state_sum_obs_num_t/((T-1))

            sigma_estimate[f] = sigma_single_estimate
            
        sigma_list.append(sigma_estimate)    
        
        GraphFilterSmoother.mu_0   = mu_0_list[it]
        GraphFilterSmoother.transP = transP_list[it]
        GraphFilterSmoother.c      = c_list[it]
        GraphFilterSmoother.sigma  = sigma_list[it]

        if it%10==0:
            OUTPUT = dict({'mu_0_list': mu_0_list, 'transP_list': transP_list, 'c_list': c_list, 'sigma_list': sigma_list}) 
    
            ########################################################################
            # Save the results
            ########################################################################
            ########################################################################
            file_name = "GraphFilterSmoother-VarParameters-101-"
            file_name = file_name+ "lenStateSpace-"+str(nstates)+".pkl"       
            f= open(file_name,"a")
            f.close()
            ########################################################################
           
            file = open(file_name,"wb")
            pkl.dump(OUTPUT, file)
            file.close()
            ########################################################################
 