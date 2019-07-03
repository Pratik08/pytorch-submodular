from nesterov import FacilityLocationSelection
import numpy
import time
import torch

numpy.random.seed(8)
for n in [500]:
    X = numpy.concatenate([numpy.random.normal(7.5, 1, size=(n, 2)),
                           numpy.random.normal(2, 1, size=(n, 2)),
                           numpy.random.normal(15, 1, size=(n, 2))])
    start_time = time.time()
    fl = FacilityLocationSelection(n, 'euclidean', n_greedy_samples=1)
    fl.fit(X)
    end_time = time.time()
    print("n = ", str(n)," time: " + str(end_time - start_time))
    print(fl.ranking)
