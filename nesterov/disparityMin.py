# disparityMin.py
# Author: Suraj Kothawade <suraj.kothawade@usc.edu>
# Date: 20th July 2019

import torch
import tqdm
import numpy

from .base import SubmodularSelection


def select_next(X, gains, current_values, mask):
    for idx in range(X.shape[0]):
        if mask[idx] == 1:
            continue

        a = torch.min(X[idx], current_values)
        gains[idx] = (a + current_values).sum()

    return torch.argmax(gains)


class DisparityMin(SubmodularSelection):
    def __init__(self, n_samples=10, pairwise_func='euclidean', n_greedy_samples=1, budget=None, cost_values=None, verbose=False):
        self.pairwise_func_name = pairwise_func
        self.budget = budget
        self.cost_values = cost_values
        self.current_cost = 0.
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')

        if ((not (self.cost_values is None)) and (self.budget is None)):
            raise ValueError(
                "Budget needs to be assigned when cost_values is not None.")

        if (self.budget is None):
            self.budget = n_samples

        def norm(x): return torch.sqrt(
            torch.sum((x * x), dim=1)).reshape(x.shape[0], 1)
        def norm2(x): return torch.sum((x * x), dim=1).reshape(x.shape[0], 1)

        if pairwise_func == 'corr':
            self.pairwise_func = lambda X: numpy.corrcoef(X, rowvar=True) ** 2.
        elif pairwise_func == 'cosine':
            self.pairwise_func = lambda X: torch.abs(
                torch.mm(X, torch.transpose(X, 0, 1)) / torch.mm(norm(X), torch.transpose(norm(X), 0, 1)))
        elif pairwise_func == 'euclidean':
            self.pairwise_func = lambda X: -(
                torch.transpose(-2 * torch.mm(X, torch.transpose(X, 0, 1)) + norm2(X), 0, 1) + norm2(X))
        elif pairwise_func == 'precomputed':
            self.pairwise_func = pairwise_func
        elif callable(pairwise_func):
            self.pairwise_func = pairwise_func
        else:
            raise KeyError(
                "Must be one of 'euclidean', 'corr', 'cosine', 'precomputed' or a custom function.")

        super(DisparityMin, self).__init__(
            n_samples, n_greedy_samples, verbose)

    def fit(self, X, y=None):
        f = self.pairwise_func

        if (self.budget and (self.cost_values is None)):
            self.cost_values = torch.ones(
                X.shape[0], dtype=torch.float64).to(self.device)
        elif (self.budget and (not (self.cost_values is None)) and len(self.cost_values) != X.shape[0]):
            raise ValueError(
                "Invalid cost_values. A cost needs to be associated with every element of the ground set.")

        if self.verbose:
            self.pbar = tqdm(total=self.n_samples)
            self.pbar.update(1)

        if self.pairwise_func == 'precomputed':
            X_pairwise = X
        else:
            X = torch.from_numpy(X).to(self.device)
            X_pairwise = self.pairwise_func(X)
            X_pairwise = torch.div(X_pairwise, torch.max(X_pairwise).to(
                                   self.device)).to(self.device)
            X_pairwise = torch.sub(torch.ones(X_pairwise.shape, dtype=torch.float64).to(
                self.device), X_pairwise).to(self.device)
        return super(DisparityMin, self).fit(X_pairwise, y)

    def _greedy_select(self, X_pairwise):
        self.current_values = torch.zeros(X_pairwise.shape[0],
                                          dtype=torch.float64).to(self.device)
        self.current_values += torch.max(X_pairwise).to(self.device)
        """Select elements in a naive greedy manner."""
        for i in range(self.n_greedy_samples):
            gains = torch.zeros(X_pairwise.shape[0]).to(self.device)
            best_idx = select_next(
                X_pairwise, gains, self.current_values, self.mask)

            self.ranking.append(best_idx.item())
            self.gains.append(gains[best_idx.item()])
            self.mask[best_idx] = 1

            if self.verbose:
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

                a = torch.max(X_pairwise[idx], self.current_values).to(
                    self.device)

                gain = torch.sum(
                    torch.sub(a, self.current_values)).to(self.device)

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
