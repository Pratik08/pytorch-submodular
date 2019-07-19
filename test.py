from nesterov import FacilityLocationSelection
from nesterov import MaxMarginRelSelection
import numpy
import time
import torch
import matplotlib.pyplot as plt

numpy.random.seed(8)
for n in [25]:
    X = numpy.concatenate([numpy.random.normal(7.5, 1, size=(n, 2)),
                           numpy.random.normal(2, 1, size=(n, 2)),
                           numpy.random.normal(15, 1, size=(n, 2))])

    X_1 = numpy.concatenate([numpy.random.normal(7.5, 1, size=(n, 2)),
                           numpy.random.rand(n, 2),
                           numpy.random.normal(15, 1, size=(n, 2))])

    X_2 = numpy.concatenate([numpy.random.normal(7.5, 1, size=(n, 2)),
                           numpy.array([[5, 16]]),
                           numpy.random.normal(2, 1, size=(n, 2)),
                           numpy.array([[16, 2], [12, 4]]),
                           numpy.random.normal(15, 1, size=(n, 2)),
                           numpy.array([[1, 12], [8, 8]])])

    start_time = time.time()
    # func = FacilityLocationSelection(5, 'euclidean', n_greedy_samples=1)
    func = MaxMarginRelSelection(5, 'euclidean')
    func.fit(X)
    end_time = time.time()
    print("n = ", str(n), " time: " + str(end_time - start_time))
    print(func.ranking)

    plt_x = []
    plt_y = []
    out_x = []
    out_y = []
    # out_x = [5, 16, 12, 1, 8]
    # out_y = [16, 2, 4, 12, 8]

    for idx in range(X.shape[0]):
        if idx not in func.ranking:
            plt_x.append(X[idx][0])
            plt_y.append(X[idx][1])
        else:
            out_x.append(X[idx][0])
            out_y.append(X[idx][1])

    plt.scatter(plt_x, plt_y, color='blue')
    plt.scatter(out_x, out_y, color='red')
    plt.show()
