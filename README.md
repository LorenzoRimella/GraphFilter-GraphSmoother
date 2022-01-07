# GraphFilter-GraphSmoother in Python
Graph Filter and Graph Smoother are algorithms for approximate filtering and smoothing in high-dimensional factorial hidden Markov models. The approximation involves discarding, in a principled way, likelihood factors according a notion of locality in a factor graph associated with the emission distribution.

The two versions of the algorithm are reported in two different folders.

- version 1: it is the older version of the algorithm, case specific on the "Gaussian chain example".

- version 2: it is the most recent version of the algorithm, this includes a more general version of the algorithm, applicable to any Networkx factor graph, and a Cython implementation for speed.

Remark that version 2 is used only in the London's tube experiment.

In addition this repository contains an online appendix which displays all the proofs of the theoretical results and provides additinal information about the work.