# Libraries
from cpython cimport *
import array
cimport cython
from libc.math cimport exp, log, sqrt, M_PI
from libc.stdlib cimport malloc, free
import numpy as nppy
cimport numpy as np

ctypedef np.int_t DTYPE_int_t
ctypedef np.float64_t DTYPE_float64_t

DTYPE_int_c = nppy.intc

# Power function for integers
cdef int int_pow(int base, int exponent):

    cdef int result = 1
    cdef Py_ssize_t i

    for i in range(exponent):
        result = result*base
    return result

# Gaussian emission in c
#cdef double cemission( double y, double mean, double variance):

#    cdef double out 
    
#    out = (1/(sqrt(2*M_PI*variance)))*exp(-((y-mean)*(y-mean))/(2*variance))

#    return  out

cdef double cemission( double y, double lamb):

    cdef double out 
    
    out = pow(lamb, y)*exp(-lamb)

    return  out

cdef double log_factorial( double y):

    if y ==0:
        return 0
    else:
        return log(y)+log_factorial( y-1 )

cdef double log_cemission( double y, double lamb):

    cdef double out 
    
    out = y*log(lamb) - lamb - log_factorial(y)

    return  out

# Find state and indexing of the state
cdef list find_state( np.ndarray[DTYPE_int_t, ndim=1] states, int dimension, int k):

    cdef int *state = <int *> malloc(dimension * sizeof(int))
    cdef int n_states = states.shape[0]
    
    cdef int itcomponents = dimension
    cdef Py_ssize_t i
    cdef int j = k

    cdef int index

    try:
        for i in range(0, dimension):
            index =  (j)/int_pow(n_states, (itcomponents-1)) 
            state[i] = states[index]
            j = j - index*(int_pow(n_states, (itcomponents-1)))
            itcomponents = itcomponents -1

        return [x for x in state[:dimension]]
    
    finally:
        free(state)    

cdef list find_state_index( int n_states, int dimension, int k):
    
    cdef int *state = <int *> malloc(dimension * sizeof(int))
    
    cdef int itcomponents = dimension
    cdef Py_ssize_t i
    cdef int j = k

    try:
    
        for i in range(0, dimension):
            index = ((j)/int_pow(n_states, (itcomponents-1)))
            state[i] = index
            j = j - index*(int_pow( n_states, (itcomponents-1)))
            itcomponents = itcomponents -1
        
        return [x for x in state[:dimension]]

    finally:
        free(state)

# Find the location of an element from a set of elements
cdef int location(int elem, np.ndarray[DTYPE_int_t, ndim=1] set_elem):

    cdef int i=0

    while set_elem[i]!=elem:

        i = i+1

    return i

# Scalar product
cdef double cscalar( list vec1, np.ndarray[DTYPE_float64_t, ndim=1] vec2):

    cdef int evolving_dimension = vec2.shape[0] 
    cdef double vec_prod 
    cdef Py_ssize_t i

    vec_prod = 0

    for i in range(0, evolving_dimension):

        vec_prod = vec_prod + vec1[i]*vec2[i]

    return vec_prod

# Dot product in C
cdef list cdot( list vec, np.ndarray[DTYPE_float64_t, ndim=2] matrix):

    cdef int evolving_dimension = matrix.shape[0] 
    cdef double *vec_matrix = <double *> malloc(evolving_dimension * sizeof(double))
    cdef Py_ssize_t i, j

    try:
        for j in range(0, evolving_dimension):

            vec_matrix[j] = 0

            for i in range(0, evolving_dimension):

                vec_matrix[j] = vec_matrix[j] + vec[i]*matrix[i,j]

        return [x for x in vec_matrix[:evolving_dimension]]

    finally:
        free(vec_matrix)

# Divide each element of a List by a constant
cdef list divide(int list_size, list List, double constant):

    cdef list output = []

    cdef Py_ssize_t i

    for i in range(0, list_size):
        if List[i]!=0:
            output.append(List[i]/constant)
        else:
            output.append(0)

    return output

cdef list exp_divide(int list_size, list List, double constant):

    cdef list output = []

    cdef Py_ssize_t i

    for i in range(0, list_size):
        
        output.append(exp(List[i])/constant)

    return output

# Function to marginilize out components outside K
cdef list marginalization(int partition_element, int n_states, list vec, np.ndarray[DTYPE_int_t, ndim=1] N2m2_K):

    cdef int size = 1
    cdef int elem_index
    cdef double *marginalized_vec = <double *> malloc( n_states*sizeof(double) )
    cdef Py_ssize_t i, j, indexwhich

    try:
        for i in range(0, n_states):
            marginalized_vec[i] = 0

        for j in range(0, int_pow( n_states, N2m2_K.shape[0])):

            current = find_state_index( n_states, N2m2_K.shape[0], j)

            position=0
            for indexwhich in range(0, size):
                elem_index = location(partition_element, N2m2_K)
                position = position+ current[elem_index]*( int_pow( n_states, (size-indexwhich-1)))

            marginalized_vec[position]= marginalized_vec[position]+ vec[j]

        return [x for x in marginalized_vec[:n_states]]

    finally:
        free(marginalized_vec)

# Filtering algorithm
def filtering(int partition_length, np.ndarray[DTYPE_int_t, ndim=1] states, int T, list YT, list mu_0, list transP, list lamb, list N2m1, list N2m2, list N1F):

    cdef list filtering = []
    cdef list filtering_t
    cdef list filtering_t_N2m2K
    cdef list log_filtering_t_N2m2K

    cdef list state, state_index

    cdef int n_states = states.shape[0]
    cdef int n_components, n_factors

    cdef int state_v_index, v
    cdef double Ppi_v_state

    cdef int f, n_N2m1_f
    cdef double y_t_f, mu_t_f, sigma_t_f

    cdef Py_ssize_t t, K, state_number, v_index, f_index, v_prime_index

    # Append the initial distribution
    filtering.append(mu_0)

    # Cycle in time
    for t in range(1, T):
        filtering_t = []

        # Cycle over the partition elements
        for K in range(0, partition_length):
            filtering_t_N2m2K = []
            log_filtering_t_N2m2K = []

            n_components = N2m2[K].shape[0]
            n_factors    = N2m1[K].shape[0]

            # Create a local approximation of the joint: first step of localization
            normalizingconst = 0
            for state_number in range(0, int_pow(n_states, n_components) ):
                # Initial condition of filtering_t_K[K][state_number]
                # filtering_t_N2m2K.append(1)
                log_filtering_t_N2m2K.append(0)

                state       = find_state( states, n_components, state_number)
                state_index = find_state_index( n_states, n_components, state_number)

                # Kernel operator 
                for v_index in range(0, n_components):

                    state_v_index = state_index[v_index]
                    v = N2m2[K][v_index]
                    Ppi_v_state = cscalar( filtering[t-1][v], transP[v][:, state_v_index] )

                    # filtering_t_N2m2K[state_number] = filtering_t_N2m2K[state_number]*Ppi_v_state
                    log_filtering_t_N2m2K[state_number] = log_filtering_t_N2m2K[state_number] + log(Ppi_v_state)

                    # if Ppi_v_state !=0:
                    #     log_filtering_t_N2m2K[state_number] = log_filtering_t_N2m2K[state_number] + log(Ppi_v_state)
                    # else:
                    #     log_filtering_t_N2m2K[state_number] = None

                # Approximate correction operator
                for f_index in range(0, n_factors):
                    f = N2m1[K][f_index]
                    n_N2m1_f = N1F[f].shape[0]

                    mean = 0
                    for v_prime_index in range(0, n_N2m1_f): 

                        elem_index = location( N1F[f][v_prime_index], N2m2[K] )
                        mean += state[elem_index]

                    y_t_f     = YT[t][f]
                    lamb_t_f    = lamb[f]*mean
                    
                    # filtering_t_N2m2K[state_number] = filtering_t_N2m2K[state_number]*cemission( y_t_f, lamb_t_f ) 
                    if lamb_t_f!=0:
                        log_filtering_t_N2m2K[state_number] = log_filtering_t_N2m2K[state_number] + log_cemission( y_t_f, lamb_t_f ) 

                    else:
                        if y_t_f!=0:
                            log_filtering_t_N2m2K[state_number] = log(0)

                        else:
                            log_filtering_t_N2m2K[state_number] = log_filtering_t_N2m2K[state_number]

                    # if log_filtering_t_N2m2K[state_number] != None:

                    #     if lamb_t_f!=0:
                    #         log_filtering_t_N2m2K[state_number] = log_filtering_t_N2m2K[state_number] + log_cemission( y_t_f, lamb_t_f )
                    #     else:
                    #         log_filtering_t_N2m2K[state_number] = None

                
                # Update the total sum
                # normalizingconst = normalizingconst + filtering_t_N2m2K[state_number]
                normalizingconst = normalizingconst + exp(log_filtering_t_N2m2K[state_number])

                # if log_filtering_t_N2m2K[state_number] != None:
                #     normalizingconst = normalizingconst + exp(log_filtering_t_N2m2K[state_number])

            # Normalize the resulting distribution
            # filtering_t_N2m2K = divide( int_pow(n_states, n_components), filtering_t_N2m2K, normalizingconst )

            filtering_t_N2m2K = exp_divide( int_pow(n_states, n_components), log_filtering_t_N2m2K, normalizingconst )

            # Append the marginalized distribution: second step of localization 
            filtering_t.append( marginalization( K, n_states, filtering_t_N2m2K, N2m2[K] ) )

        # Append the resulting filtering
        filtering.append(filtering_t)

    return filtering

# 1 step Filtering algorithm
def one_step_filtering(int partition_length, np.ndarray[DTYPE_int_t, ndim=1] states, list YT, list mu_0, list transP, list lamb, list N2m1, list N2m2, list N1F):

    cdef list filtering = []
    cdef list filtering_t
    cdef list filtering_t_N2m2K
    cdef list log_filtering_t_N2m2K

    cdef list state, state_index

    cdef int n_states = states.shape[0]
    cdef int n_components, n_factors

    cdef int state_v_index, v
    cdef double Ppi_v_state

    cdef int f, n_N2m1_f
    cdef double y_t_f, mu_t_f, sigma_t_f

    cdef Py_ssize_t t, K, state_number, v_index, f_index, v_prime_index

    # Append the initial distribution
    filtering.append(mu_0)

    # Cycle in time
    for t in range(1, 2):
        filtering_t = []

        # Cycle over the partition elements
        for K in range(0, partition_length):
            filtering_t_N2m2K = []
            log_filtering_t_N2m2K = []

            n_components = N2m2[K].shape[0]
            n_factors    = N2m1[K].shape[0]

            # Create a local approximation of the joint: first step of localization
            normalizingconst = 0
            for state_number in range(0, int_pow(n_states, n_components) ):
                # Initial condition of filtering_t_K[K][state_number]
                # filtering_t_N2m2K.append(1)
                log_filtering_t_N2m2K.append(0)

                state       = find_state( states, n_components, state_number)
                state_index = find_state_index( n_states, n_components, state_number)

                # Kernel operator 
                for v_index in range(0, n_components):

                    state_v_index = state_index[v_index]
                    v = N2m2[K][v_index]
                    Ppi_v_state = cscalar( filtering[t-1][v], transP[v][:, state_v_index] )

                    # filtering_t_N2m2K[state_number] = filtering_t_N2m2K[state_number]*Ppi_v_state
                    log_filtering_t_N2m2K[state_number] = log_filtering_t_N2m2K[state_number] + log(Ppi_v_state)

                    # if Ppi_v_state !=0:
                    #     log_filtering_t_N2m2K[state_number] = log_filtering_t_N2m2K[state_number] + log(Ppi_v_state)
                    # else:
                    #     log_filtering_t_N2m2K[state_number] = None

                # Approximate correction operator
                for f_index in range(0, n_factors):
                    f = N2m1[K][f_index]
                    n_N2m1_f = N1F[f].shape[0]

                    mean = 0
                    for v_prime_index in range(0, n_N2m1_f): 

                        elem_index = location( N1F[f][v_prime_index], N2m2[K] )
                        mean += state[elem_index]

                    y_t_f     = YT[t-1][f]
                    lamb_t_f    = lamb[f]*mean
                    
                    # filtering_t_N2m2K[state_number] = filtering_t_N2m2K[state_number]*cemission( y_t_f, lamb_t_f ) 
                    if lamb_t_f!=0:
                        log_filtering_t_N2m2K[state_number] = log_filtering_t_N2m2K[state_number] + log_cemission( y_t_f, lamb_t_f ) 

                    else:
                        if y_t_f!=0:
                            log_filtering_t_N2m2K[state_number] = log(0)

                        else:
                            log_filtering_t_N2m2K[state_number] = log_filtering_t_N2m2K[state_number]

                    # if log_filtering_t_N2m2K[state_number] != None:

                    #     if lamb_t_f!=0:
                    #         log_filtering_t_N2m2K[state_number] = log_filtering_t_N2m2K[state_number] + log_cemission( y_t_f, lamb_t_f )
                    #     else:
                    #         log_filtering_t_N2m2K[state_number] = None

                
                # Update the total sum
                # normalizingconst = normalizingconst + filtering_t_N2m2K[state_number]
                normalizingconst = normalizingconst + exp(log_filtering_t_N2m2K[state_number])

                # if log_filtering_t_N2m2K[state_number] != None:
                #     normalizingconst = normalizingconst + exp(log_filtering_t_N2m2K[state_number])

            # Normalize the resulting distribution
            # filtering_t_N2m2K = divide( int_pow(n_states, n_components), filtering_t_N2m2K, normalizingconst )

            filtering_t_N2m2K = exp_divide( int_pow(n_states, n_components), log_filtering_t_N2m2K, normalizingconst )

            # Append the marginalized distribution: second step of localization 
            filtering_t.append( marginalization( K, n_states, filtering_t_N2m2K, N2m2[K] ) )

        # Append the resulting filtering
        filtering.append(filtering_t)

    return filtering


# Smoothing algorithm
def smoothing(int partition_length, int T, list P2m2, list pifiltering):
    
    cdef list pismoother = []
    cdef list newsmoother
    cdef np.ndarray[DTYPE_float64_t, ndim=2] kernel, support_kernel
    cdef double normalization
    cdef Py_ssize_t t, i, j

    for t in range(0, T):
        pismoother.append(None)

    pismoother[T-1] = pifiltering[T-1]
    
    for t in range(1, T):                
       # print(t)
        newsmoother = []
        for K in range(0, partition_length):

            kernel = nppy.copy(P2m2[K])
            support_kernel = nppy.copy(P2m2[K])

                      
            for i in range(kernel.shape[0]):

                normalization = 0
                for j in range(kernel.shape[1]):
                  #  print(i,j)
                    normalization = normalization + pifiltering[T-t-1][K][j]*(kernel[j,i])

                for j in range(kernel.shape[1]):
                    if normalization!=0:
                        support_kernel[i, j] = kernel[j,i]*pifiltering[T-t-1][K][j]/normalization

                    else:
                        support_kernel[i, j] =0
            
            newsmoother.append(cdot(pismoother[T-t][K],  support_kernel))
            
        pismoother[T-1-t] = newsmoother
        
    return pismoother








   