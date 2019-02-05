#Required files
import numpy as np

import GraphFilter_GraphSmoother

import GharamaniJordan

import EMalgorithmtoy

import time

import pickle


# model properties
nlayers = 5
nstates = 2
nobservations = 4


# true initial
mu0true = np.array([1,0], dtype= float)

mu0true2 = np.zeros((nlayers,nstates), dtype = float)
for i in range(0, nlayers):
    mu0true2[i,:] = mu0true

# true kernel
Ptrue = np.array([[ 0.40, 0.60],
                  [ 0.80, 0.20]], dtype = float)

transPtrue = np.zeros((nlayers,nstates,nstates), dtype = float)
for i in range(0, nlayers):
        transPtrue[i,:,:] = Ptrue



# true c
ctrue = 2

Wtrue = np.array([[np.array(range(nstates), dtype = float)+1],
              [np.array(range(nstates), dtype = float)+1]], dtype=float)

for ite in range(2, nlayers):
    # matrix W
    newWtrue = np.zeros((ite+1,ite,nstates), dtype=float)
    newWtrue[0:ite, 0:ite-1, :] = Wtrue
    newWtrue[ite-1, ite-1, :] = Wtrue[0, 0, :]
    newWtrue[ite, ite-1, :]   = Wtrue[0, 0, :]

    Wtrue = newWtrue

Wtrue = ctrue*Wtrue


# true sigma
sigmatrue = float(4)

Ctrue = sigmatrue*np.diag(np.ones(nobservations))

# time
T= 200


# simulate data
np.random.seed(567)
dicto = GraphFilter_GraphSmoother.FHMM(T+1, nstates, nlayers, nobservations, mu0true2, transPtrue, Wtrue, Ctrue)

YT= dicto['observations']


# partition choice
trivialpart = list()

for i in range(0,nlayers):
    trivialpart.append(['H'+str(i)])



############################
# EM algorithm

EMdict = {}

# initial conditions
initial0 = list()

P0 = list()

c0 = np.linspace(0.25,5,20)

sigma0 = np.linspace(0.5,8,20)



for i in np.linspace(-9,9,20):
    mu0new = np.array([0.5-5*(i/100),0.5+5*(i/100)], dtype=float)
    initial0.append(mu0new)
    
    Pnew = np.array([[.5-5*(i/100), .5+5*(i/100)],
                     [.5+5*(i/100), .5-5*(i/100)]], dtype=float)
    P0.append(Pnew)

for i in range(-5,5):
    mu0new = np.array([0.5-5*(i/100),0.5+5*(i/100)], dtype=float)
    initial0.append(mu0new)
    
    Pnew = np.array([[.5-5*(i/100), .5+5*(i/100)],
                     [.5+5*(i/100), .5-5*(i/100)]], dtype=float)
    P0.append(Pnew)
    
m=0

# EM for different initial conditions
for value in range(0,20):
    print(value)
    
    EMdict['cond'+str(value)] = EMalgorithmtoy.EMtoy(nlayers, YT, initial0[value], P0[value], c0[value], sigma0[value], trivialpart, m, 200)



##########################################
# FILES
# save the files in pickle format
EMf = open("EM1.pkl","wb")
pickle.dump(EMdict,EMf)
EMf.close()

# to open the file use:
#
# with open("Data/EM1.pkl", "rb") as handle:
#    EM1 = pickle.load(handle, encoding='latin1')
#
# In this way you create a Python dictonaries. Use .keys() to see 
# the keys and EM1['key'] to select a precise key.

