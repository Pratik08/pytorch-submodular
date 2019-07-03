# base.py
# Author: Pratik Dubal <pratik.dubal@columbia.edu>
# Date: 2nd July 2019

import torch
import tqdm
import numpy

from .utils import PriorityQueue

class SubmodularSelection(object):

	def __init__(self, n_samples, n_greedy_samples=1, verbose=False):
		self.pq = PriorityQueue()
		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

		self.n_samples = n_samples
		self.n_greedy_samples = n_greedy_samples
		self.verbose = verbose
		self.ranking = None
		self.gains = None

	def fit(self, X, y=None):

		if self.verbose == True:
			self.pbar = tqdm(total=self.n_samples)

		self.current_values = torch.zeros(X.shape[0], dtype=torch.float64).to(self.device)
		self.current_values += torch.min(X).to(self.device)
		self.mask = torch.zeros(X.shape[0], dtype=torch.int8).to(self.device)
		self.ranking = []
		self.gains = []

		# Select using the greedy algorithm first returning the gains from
		# the last round of selection.
		gains = self._greedy_select(X)

		# Populate the priority queue following greedy selection
		if self.n_greedy_samples < self.n_samples:
			for idx, gain in enumerate(gains):
				if self.mask[idx].item() != 1:
					self.pq.add(idx, -gain.item())

			# Now select remaining elements using the lazy greedy algorithm.
			self._lazy_greedy_select(X)

		if self.verbose == True:
			self.pbar.close()

		self.ranking = numpy.array(self.ranking)
		return self

	def _greedy_select(self, X):
		raise NotImplementedError

	def _lazy_greedy_select(self, X):
		raise NotImplementedError

	# def transform(self, X, y=None):
	# 	if y is not None:
	# 		return X[self.ranking], y[self.ranking]
	# 	return X[self.ranking]
    #
	# def fit_transform(self, X, y=None):
	# 	return self.fit(X, y).transform(X, y)
