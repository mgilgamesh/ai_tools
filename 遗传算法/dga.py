import numpy as np
from geneticalgorithm import geneticalgorithm as ga


def f(X):
    return np.sum(X)


varbound = np.array([[0.5, 1.5], [1, 100], [0, 1]])
vartype = np.array([['real'], ['int'], ['int']])
model = ga(function=f, dimension=3, variable_type_mixed=vartype, variable_boundaries=varbound)

model.run()