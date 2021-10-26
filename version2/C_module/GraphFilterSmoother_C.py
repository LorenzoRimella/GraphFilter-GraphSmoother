import C_GraphFilterSmoother
import numpy as np
import networkx as nx
from networkx.algorithms import bipartite

# Define a GraphFilterSmoother object
class GraphFilterSmoother_C:

    def __init__(self, factor_graph, states, m, partition, mu_0, transP, lamb):

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

        if len(partition["0"])!=1:
            print("The partition should be on singleton.")

        # set the intial conditions
        # for the kernel
        self.mu_0 = mu_0
        self.transP = transP
        #for the emission distribution
        self.lamb = lamb

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
    
    # 
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
        
        N2m1 = []

        for K in range(0, len(self.partition)):
            supp = []
            for element in self.partition[str(K)]:
                for neighbour in self.factors.intersection(self.shortest_path_length[element].keys()):
                    supp.append(int(neighbour[1:]))
        
            N2m1.append(np.sort(np.array(supp)))

        return N2m1

    # compute the set of neighbours distant 2m+2
    def N2m2K(self):
            
        N2m2 = []

        for K in range(0, len(self.partition)):
            supp = []
            for element in self.partition[str(K)]:
                for neighbour in self.hidden_variables.intersection(self.shortest_path_length[element].keys()):
                        supp.append(int(neighbour[1:]))

            N2m2.append(np.sort(np.array(supp)))
                      
        return N2m2

    # define the neighbours of level 1 of all the factors
    def N1FK(self):

        N1F = []

        for factor_index in range(0, len(self.factors)):

            zeros = ""
            if factor_index<10:
                zeros = "0"
            factor = "F"+zeros+str(factor_index)
            N1F_factor = []
            for neighbour in self.shortest_path_length[factor].keys():

                if self.shortest_path_length[factor][neighbour] == 1:
                    N1F_factor.append(int(neighbour[1:]))

            N1F.append( np.sort(np.array(N1F_factor)) )

        return N1F
    
    def preprocessing(self):
        # define the neighbours
        self.N2m1  = self.N2m1K()
        self.N2m2  = self.N2m2K()
        self.N1F   = self.N1FK()

    def filtering(self, YT):

        return C_GraphFilterSmoother.filtering( len(self.partition), np.array(self.states), len(YT), list(YT), self.mu_0, self.transP, self.lamb, self.N2m1, self.N2m2, self.N1F )


    def one_step_filtering(self, YT, pifilt):
    
        return C_GraphFilterSmoother.one_step_filtering( len(self.partition), np.array(self.states), list(YT), pifilt, self.transP, self.lamb, self.N2m1, self.N2m2, self.N1F )

    def smoothing(self, pifiltering):

        return C_GraphFilterSmoother.smoothing( len(self.partition), len(pifiltering), self.transP, pifiltering ) 

    def expected_posterior_mean_pred(self, pifiltering):
    
        self.T = len(pifiltering)

        posterior_mean = []
        for f in range(len(self.factors)):

            posterior_mean_f = []
            posterior_mean_f.append(0)

            for t in range(0, self.T-1):

                mean = 0
                for state_number in range(len(self.states)**len(self.N1F[f])):

                    state = self.find_state(len(self.N1F[f]), state_number)
                    state_index = self.find_state_index(len(self.N1F[f]), state_number)

                    prod = 1
                    
                    for i in range(len(self.N1F[f])):

                        Ppi = np.dot( pifiltering[t][self.N1F[f][i]], self.transP[self.N1F[f][i]] )

                        prod = prod*Ppi[state_index[i]]

                    mean = mean + self.lamb[f]*np.sum(state)*prod

                posterior_mean_f.append(mean)

            posterior_mean.append(posterior_mean_f)

        return posterior_mean

    def maximum_at_posteriori_pred(self, pifiltering):
        
        self.T = len(pifiltering)

        maximum_at_posteriori = []
        for f in range(len(self.factors)):

            maximum_at_posteriori_f = []
            maximum_at_posteriori_f.append(0)

            for t in range(0, self.T-1):

                max_ = 0
                for state_number in range(len(self.states)**len(self.N1F[f])):

                    state = self.find_state(len(self.N1F[f]), state_number)
                    state_index = self.find_state_index(len(self.N1F[f]), state_number)

                    prod = 1
                    
                    for i in range(len(self.N1F[f])):

                        Ppi = np.dot( pifiltering[t][self.N1F[f][i]], self.transP[self.N1F[f][i]] )

                        prod = prod*Ppi[state_index[i]]
                        
                    if max_< prod:
                        max_ = prod
                        mean = self.lamb[f]*np.sum(state)

                maximum_at_posteriori_f.append(mean)

            maximum_at_posteriori.append(maximum_at_posteriori_f)

        return maximum_at_posteriori
    
    
    
    def posterior_predictive_sample(self, pifiltering, n_sample):
        
        self.T = len(pifiltering)
        
        y_posterior_pred = []

        for t in range(0, self.T-1):
            
            x_tp1 = np.zeros((self.M, n_sample))
            
            for v in range(self.M):
                
                Ppi = np.dot( pifiltering[t][v], self.transP[v] )
                
                sample_v = np.random.choice( self.states, size = n_sample, p = Ppi )
                
                x_tp1[v, :] = sample_v
              
            y_tp1 = np.zeros((self.n_factors, n_sample))
            
            for f in range(self.n_factors):
            
                poisson_param = np.zeros(n_sample)
                
                for i in range(len(self.N1F[f])): 
                    
                    poisson_param = poisson_param + x_tp1[self.N1F[f][i], :]
                    
                poisson_param = self.lamb[f]*poisson_param
                
                y_tp1[f, :] = np.random.poisson( poisson_param )
                
            y_posterior_pred.append(y_tp1)
            
        return y_posterior_pred
    
    
    def posterior_predictive_sample_missing_data(self, pifiltering, n_sample, start_miss, num_miss):

        self.T = len(pifiltering)

        y_posterior_pred = []

        for t in range(0, start_miss+1):

            x_tp1 = np.zeros((self.M, n_sample), dtype = int)

            for v in range(self.M):

                Ppi = np.dot( pifiltering[t][v], self.transP[v] )

                sample_v = np.random.choice( self.states, size = n_sample, p = Ppi )

                x_tp1[v, :] = sample_v

            y_tp1 = np.zeros((self.n_factors, n_sample))

            for f in range(self.n_factors):

                poisson_param = np.zeros(n_sample)

                for i in range(len(self.N1F[f])): 

                    poisson_param = poisson_param + x_tp1[self.N1F[f][i], :]

                poisson_param = self.lamb[f]*poisson_param

                y_tp1[f, :] = np.random.poisson( poisson_param )

            y_posterior_pred.append(y_tp1)

        for t in range(start_miss+1, start_miss+num_miss+1):

            for v in range(self.M):

                Ppi = self.transP[v][x_tp1[v, :], :]

                for sampling in range(n_sample):
                    sample_v_sampling = np.random.choice( self.states, size = 1, p = Ppi[sampling,:] )

                    x_tp1[v, sampling] = sample_v_sampling     

            y_tp1 = np.zeros((self.n_factors, n_sample))

            for f in range(self.n_factors):

                poisson_param = np.zeros(n_sample)

                for i in range(len(self.N1F[f])): 

                    poisson_param = poisson_param + x_tp1[self.N1F[f][i], :]

                poisson_param = self.lamb[f]*poisson_param

                y_tp1[f, :] = np.random.poisson( poisson_param )

            y_posterior_pred.append(y_tp1)


        for t in range(start_miss+num_miss+1, self.T-1):

            x_tp1 = np.zeros((self.M, n_sample))

            for v in range(self.M):

                Ppi = np.dot( pifiltering[t][v], self.transP[v] )

                sample_v = np.random.choice( self.states, size = n_sample, p = Ppi )

                x_tp1[v, :] = sample_v

            y_tp1 = np.zeros((self.n_factors, n_sample))

            for f in range(self.n_factors):

                poisson_param = np.zeros(n_sample)

                for i in range(len(self.N1F[f])): 

                    poisson_param = poisson_param + x_tp1[self.N1F[f][i], :]

                poisson_param = self.lamb[f]*poisson_param

                y_tp1[f, :] = np.random.poisson( poisson_param )

            y_posterior_pred.append(y_tp1)

        return y_posterior_pred
                    
                
                
                
                
                
                
                
                
                
                
        
        