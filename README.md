# GraphFilter-GraphSmoother
Graph Filter and Graph Smoother are algorithms for approximate filtering and smoothing in high-dimensional factorial hidden Markov models. The approximation involves discarding, in a principled way, likelihood factors according a notion of locality in a factor graph associated with the emission distribution.

In this repository you can find the following functions:

- EMalgorithmtoy.py: function for the EM algorithm with the Graph Smoother

- GJsubroutine.py: contains the forward-backward algorithm used in the partially decoupled variational Bayes algorithm (Ghahramani- Jordan)

- GraphFilter_GraphSmoother.py: contains all the functions which you need to compute the Graph Filter and the Graph Smoother

- GharamaniJordan.py: the computationally intensive forward-backward algorithm
	
- Probabilities_Forecasting_HMM.py: functions to forecast the states given the probabilities 

- VBayes.py: the completely and partially decoupled variational Bayes algorithms.
