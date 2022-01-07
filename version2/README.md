# GraphFilter-GraphSmoother in Python: version 1
Graph Filter and Graph Smoother are algorithms for approximate filtering and smoothing in high-dimensional factorial hidden Markov models. The approximation involves discarding, in a principled way, likelihood factors according a notion of locality in a factor graph associated with the emission distribution.

In this repository you can find the following files:

- Tutorial Graph Filter-Smoother.ipynb: a jupyter notebook explaining how to train the algorithm and how to reproduce the results in the paper;

- Tutorial LSTM.ipynb: a jupyter notebook explaining how to train the LSTM baseline and how to reproduce the results in the paper;

Along with the above files there are two folders:

- Data: containing all the needed data for the experiment;

- Script: containing all the important scripts.

# Oyster data: Cleaning procedure: version 2
The Oyster dataset was about the movement that people made from one station to another, and so where a person starts her/his journey and where the same person finished it. In this dataset were included useless and redundant information that we have excluded from the analysis (e.g. information on the fare, time stored in different formats). The data displayed also different modes of transport, but for sake of simplicity we have decided to drop them and focus on tube's travel. The final dataset consists of integer counts of the inflow and the outflow of passengers from Monday to Friday per station every 10 minutes from 00:00 am to 00:00 am of the next day. The data are split into training given by Monday, Tuesday and Wednesday and test consisting of Thursday and Friday. 
