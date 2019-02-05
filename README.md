# GraphFilter-GraphSmoother
Graph Filter and Graph Smoother are algorithms for approximate filtering and smoothing in high-dimensional factorial hidden Markov models. The approximation involves discarding, in a principled way, likelihood factors according a notion of locality in a factor graph associated with the emission distribution.

In this repository you can find the following Python files:

- GraphFilter_GraphSmoother.py: contains our implementation of the Graph Filter and the Graph Smoother;

- Probabilities_Forecasting_HMM.py: contains some functions to forecast the states given the probabilities vectors;

- EMalgorithmtoy.py: function for the EM algorithm with Graph Smoother in the chain case (M-dimensional hidden variable and M-1 factors);

- GharamaniJordan.py: our code for the optimal forward-backward algorithm for FHMM by Ghahramani and Jordan;

- GJsubroutine.py: the forward-backward algorithm used in the partially decoupled variational Bayes algorithm (Ghahramani and Jordan);

- VBayes.py: contains functions for the completely and partially decoupled variational Bayes algorithms;

- EM1.py: it is an example on how to use the Graph Filter-Smoother in the EM algorihtm, running EM1.py gives a .pkl file as output containing the results for a synthetic dataset (simulated in the same file) of the EM algorithm where we used different initial conditions;

- london.py: it is our application to the London's tube, the first part of the code is the cleaning procedure, where we adjust the dataset according to our aim, the second part is an EM algorithm with different initial conditions. We strongly reccomend to reduce the number of initial conditions if you are not interested in a long simulation. Remark that for this file we need the file 'Nov09JnyExport.csv' which can be downloaded at https://api-portal.tfl.gov.uk/docs under the section: Oyster card data. 

There is also a supplementary file:

- London2.pickle: this is just a .pkl file where the graph structure of London's tube is stored.
