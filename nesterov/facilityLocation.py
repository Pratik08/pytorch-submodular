# facilityLocation.py
# Author: Pratik Dubal <pratik.dubal@columbia.edu>
# Date: 2nd July 2019

import torch
import tqdm
import numpy
import time

from .base import SubmodularSelection


def select_next(X, gains, current_values, mask):
	for idx in range(X.shape[0]):
		if mask[idx] == 1:
			continue

		a = torch.max(X[idx], current_values)
		gains[idx] = (a - current_values).sum()

	return torch.argmax(gains)

def select_next_knapsack(X, gains, current_values, cost_values, mask):

	a = torch.max(torch.div(X, cost_values), current_values)
	temp = torch.sum(torch.sub(a, current_values), dim=1)
	for idx in range(X.shape[0]):
		if mask[idx].item() != 1:
			gains[idx] = temp[idx]
	return torch.argmax(gains)

class FacilityLocationSelection(SubmodularSelection):
	def __init__(self, n_samples=10, pairwise_func='euclidean', n_greedy_samples=1, budget=None, cost_values=None, verbose=False):
		self.pairwise_func_name = pairwise_func
		self.budget = budget
		self.cost_values = cost_values
		self.current_cost = 0.
		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

		if ((not (self.cost_values is None)) and (self.budget is None)):
			raise ValueError("Budget needs to be assigned when cost_values is not None.")

		if (self.budget is None):
			self.budget = n_samples

		norm = lambda x: numpy.sqrt((x*x).sum(axis=1)).reshape(x.shape[0], 1)
		norm2 = lambda x: torch.sum((x*x), dim=1).reshape(x.shape[0], 1)

		if pairwise_func == 'corr':
			self.pairwise_func = lambda X: numpy.corrcoef(X, rowvar=True) ** 2.
		elif pairwise_func == 'cosine':
			self.pairwise_func = lambda X: numpy.abs(numpy.dot(X, X.T) / (norm(X).dot(norm(X).T)))
		elif pairwise_func == 'euclidean':
			self.pairwise_func = lambda X: -(torch.transpose(-2 * torch.mm(X, torch.transpose(X, 0, 1)) + norm2(X), 0, 1) + norm2(X))
		elif pairwise_func == 'precomputed':
			self.pairwise_func = pairwise_func
		elif callable(pairwise_func):
			self.pairwise_func = pairwise_func
		else:
			raise KeyError("Must be one of 'euclidean', 'corr', 'cosine', 'precomputed'" \
				" or a custom function.")

		super(FacilityLocationSelection, self).__init__(n_samples,
			n_greedy_samples, verbose)

	def fit(self, X, y=None):
		f = self.pairwise_func

		if (self.budget and (self.cost_values is None)):
			self.cost_values = torch.ones(X.shape[0], dtype=torch.float64).to(self.device)
		elif (self.budget and (not (self.cost_values is None)) and len(self.cost_values) != X.shape[0]):
			raise ValueError("Invalid cost_values. A cost needs to be associated with every element of the ground set.")

		if self.verbose == True:
			self.pbar = tqdm(total=self.n_samples)
			self.pbar.update(1)

		if self.pairwise_func == 'precomputed':
			X_pairwise = X
		else:
			X = torch.from_numpy(X).to(self.device)
			X_pairwise = self.pairwise_func(X)

			if self.pairwise_func_name == 'euclidean':
				eps = torch.max(torch.diag(X_pairwise).to(self.device)).to(self.device)
				X_pairwise = torch.sub(X_pairwise, torch.eye(X.shape[0]).double() * eps.double()).to(self.device)
		return super(FacilityLocationSelection, self).fit(X_pairwise, y)

	def _greedy_select(self, X_pairwise):
		"""Select elements in a naive greedy manner."""

		for i in range(self.n_greedy_samples):
			gains = torch.zeros(X_pairwise.shape[0]).to(self.device)

			best_idx = select_next(X_pairwise, gains, self.current_values,
				self.mask)
			self.current_values = torch.max(X_pairwise[best_idx],
				self.current_values).to(self.device)

			self.ranking.append(best_idx.item())
			self.gains.append(gains[best_idx.item()])
			self.mask[best_idx] = 1

			if self.verbose == True:
				self.pbar.update(1)

		return gains

	def _lazy_greedy_select(self, X_pairwise):
		"""Select elements from a dense matrix in a lazy greedy manner."""

		for i in range(self.n_greedy_samples, self.n_samples):
			best_gain = 0.
			best_idx = None

			while True:
				prev_gain, idx = self.pq.pop()
				prev_gain = -prev_gain

				if best_gain >= prev_gain:
					self.pq.add(idx, -prev_gain)
					self.pq.remove(best_idx)
					break

				a = torch.max(X_pairwise[idx], self.current_values).to(self.device)

				gain = torch.sum(torch.sub(a, self.current_values)).to(self.device)

				self.pq.add(idx, -gain.item())

				if gain.item() > best_gain:
					best_gain = gain.item()
					best_idx = idx

			self.ranking.append(best_idx)
			self.gains.append(best_gain)
			self.mask[best_idx] = True

			self.current_values = torch.max(X_pairwise[best_idx],
				self.current_values).to(self.device)

			if self.verbose == True:
				self.pbar.update(1)

	def _greedy_select_knapsack(self, X_pairwise):
		"""Select elements in a naive greedy manner."""

		self.current_values = torch.div(self.current_values, self.cost_values).to(self.device)

		for i in range(self.n_greedy_samples):
			gains = torch.zeros(X_pairwise.shape[0]).to(self.device)

			best_idx = select_next_knapsack(X_pairwise, gains, self.current_values,
											self.cost_values, self.mask)

			if (self.current_cost + self.cost_values[best_idx].item() <= self.budget):
				self.ranking.append(best_idx.item())
				self.gains.append(gains[best_idx.item()])
				self.mask[best_idx.item()] = 1
				self.current_values = torch.max(torch.div(X_pairwise[best_idx.item()],
					self.cost_values).to(self.device), self.current_values).to(self.device)
				self.current_cost += self.cost_values[best_idx.item()].item()
			else:
				# might need to be changed. Updated gain is being returned.
				# Need to return gain of previous iteration.
				return gains

			self.current_values = torch.max(X_pairwise[best_idx],
				self.current_values).to(self.device)

			if self.verbose == True:
				self.pbar.update(1)

		return gains

	def _lazy_greedy_select_knapsack(self, X_pairwise):
		"""Select elements from a dense matrix in a lazy greedy manner."""

		for i in range(self.n_greedy_samples, self.n_samples):
			best_gain = 0.
			best_idx = None

			while True:
				prev_gain, idx = self.pq.pop()
				prev_gain = -prev_gain

				if best_gain >= prev_gain:
					self.pq.add(idx, -prev_gain)
					self.pq.remove(best_idx)
					break

				a = torch.max(torch.div(X_pairwise[:, idx], self.cost_values).to(self.device),
					self.current_values).to(self.device)

				gain = torch.sum(torch.sub(a, self.current_values)).to(self.device).item()

				self.pq.add(idx, -gain)

				if gain > best_gain:
					best_gain = gain
					best_idx = idx

			if (self.current_cost + self.cost_values[best_idx].item() <= self.budget):

				self.ranking.append(best_idx)
				self.gains.append(best_gain)
				self.mask[best_idx] = True
				self.current_cost += self.cost_values[best_idx.item()].item()

				self.current_values = torch.max(torch.div(X_pairwise[best_idx], self.cost_values).to(self.device),
					self.current_values).to(self.device)
			else:
				break

			if self.verbose == True:
				self.pbar.update(1)
