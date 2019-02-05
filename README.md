# GraphFilter-GraphSmoother
Graph Filter and Graph Smoother are algorithms for approximate filtering and smoothing in high-dimensional factorial hidden Markov models. The approximation involves discarding, in a principled way, likelihood factors according a notion of locality in a factor graph associated with the emission distribution.

In this repository you can find the following Python files:

- GraphFilter_GraphSmoother.py: contains our implementation of the Graph Filter and the Graph Smoother;

- Probabilities_Forecasting_HMM.py: contains some functions to forecast the states given the probabilities vectors;

- EMalgorithmtoy.py: function for the EM algorithm with Graph Smoother in the chain case ($M$-dimensional hidden variable and $M-1$ factors);

- GharamaniJordan.py: our code for the optimal forward-backward algorithm for FHMM by Ghahramani and Jordan;

- GJsubroutine.py: the forward-backward algorithm used in the partially decoupled variational Bayes algorithm (Ghahramani and Jordan)

- VBayes.py: contains functions for the completely and partially decoupled variational Bayes algorithms.
