import numpy as np
import torch
import torch.nn as nn

from abc import ABC, abstractmethod

def euclidean_distances(x, y):
    return torch.norm(x.unsqueeze(1) - y.unsqueeze(0), dim=2)


def gaussian_kernel(x, y, sigma):
    return torch.exp(- .5 / (sigma ** 2) * euclidean_distances(x, y) ** 2)


class MeanShift(ABC):

    def __init__(self, num_seeds=100, max_iters=10, epsilon=1e-2,
                 h=1., batch_size=None):
        self.num_seeds = num_seeds
        self.max_iters = max_iters
        self.epsilon = epsilon
        self.h = h
        if batch_size is None:
            batch_size = 1000
        self.batch_size = batch_size

        self.distance = None

        self.kernel = None

    def connected_components(self, Z):

        n, d = Z.shape
        K = 0

        cluster_labels = torch.ones((n,), dtype=torch.long, device=Z.device) * -1
        for i in range(n):
            if cluster_labels[i] == -1:

                distances = self.distance(Z, Z[i:i + 1])
                component_seeds = distances[:, 0] <= self.epsilon

                if torch.unique(cluster_labels[component_seeds]).shape[0] > 1:
                    temp = cluster_labels[component_seeds]
                    temp = temp[temp != -1]
                    label = torch.mode(temp)[0]
                else:
                    label = torch.tensor(K).cuda()
                    K += 1
                cluster_labels[component_seeds] = label

        return cluster_labels

    def seed_hill_climbing(self, X, Z):
        n, d = X.shape
        m = Z.shape[0]

        for _iter in range(self.max_iters):
            new_Z = Z.clone()
            for i in range(0, m, self.batch_size):
                W = self.kernel(Z[i:i + self.batch_size], X, self.h)
                Q = W / W.sum(dim=1, keepdim=True)
                new_Z[i:i + self.batch_size] = torch.mm(Q, X)

            Z = new_Z

        return Z

    def select_smart_seeds(self, X):
        n, d = X.shape

        selected_indices = -1 * torch.ones(self.num_seeds, dtype=torch.long)
        seeds = torch.empty((self.num_seeds, d), device=X.device)
        num_chosen_seeds = 0
        distances = torch.empty((n, self.num_seeds), device=X.device)
        selected_seed_index = np.random.randint(0, n)
        selected_indices[0] = selected_seed_index
        selected_seed = X[selected_seed_index, :]
        seeds[0, :] = selected_seed

        distances[:, 0] = self.distance(X, selected_seed.unsqueeze(0))[:, 0]
        num_chosen_seeds += 1

        for i in range(num_chosen_seeds, min(self.num_seeds, n)):
            distance_to_nearest_seed = torch.min(distances[:, :i], dim=1)[0]
            selected_seed_index = torch.multinomial(distance_to_nearest_seed, 1)
            selected_indices[i] = selected_seed_index
            selected_seed = torch.index_select(X, 0, selected_seed_index)[0, :]
            seeds[i, :] = selected_seed
            distances[:, i] = self.distance(X, selected_seed.unsqueeze(0))[:, 0]

        return seeds

    def mean_shift_with_seeds(self, X, Z):
        Z = self.seed_hill_climbing(X, Z)
        cluster_labels = self.connected_components(Z)

        return cluster_labels, Z

    @abstractmethod
    def mean_shift_smart_init(self):
        pass


class GaussianMeanShift(MeanShift):

    def __init__(self, num_seeds=100, max_iters=10, epsilon=0.05,
                 sigma=1.0, subsample_factor=1, batch_size=None):
        super().__init__(num_seeds=num_seeds,
                         max_iters=max_iters,
                         epsilon=epsilon,
                         h=sigma,
                         batch_size=batch_size)
        self.subsample_factor = subsample_factor
        self.distance = euclidean_distances
        self.kernel = gaussian_kernel

    def mean_shift_smart_init(self, X, sigmas=None):
        subsampled_X = X[::self.subsample_factor, ...]
        if sigmas is not None:
            subsampled_sigmas = sigmas[::self.subsample_factor]
            self.h = subsampled_sigmas.unsqueeze(0)
        seeds = self.select_smart_seeds(subsampled_X)

        seed_cluster_labels, updated_seeds = self.mean_shift_with_seeds(subsampled_X, seeds)
        distances = self.distance(X, updated_seeds)

        closest_seed_indices = torch.argmin(distances, dim=1)
        cluster_labels = seed_cluster_labels[closest_seed_indices]

        uniq_labels = torch.unique(seed_cluster_labels)
        uniq_cluster_centers = torch.zeros((uniq_labels.shape[0], updated_seeds.shape[1]), dtype=torch.float,
                                           device=updated_seeds.device)
        for i, label in enumerate(uniq_labels):
            uniq_cluster_centers[i, :] = updated_seeds[seed_cluster_labels == i, :].mean(dim=0)
        self.uniq_cluster_centers = uniq_cluster_centers
        self.uniq_labels = uniq_labels

        return cluster_labels.to(X.device)


class WeightedLoss(nn.Module):

    def __init__(self):
        super(WeightedLoss, self).__init__()
        self.weighted = False

    def generate_weight_mask(self, mask, to_ignore=None):
        N = mask.shape[0]
        if self.weighted:
            weight_mask = torch.zeros_like(mask).float()

            for i in range(N):
                unique_object_labels = torch.unique(mask[i])
                for obj in unique_object_labels:

                    if to_ignore is not None and obj in to_ignore:
                        continue

                    num_pixels = torch.sum(mask[i] == obj, dtype=torch.float)
                    weight_mask[i, mask[i] == obj] = 1 / num_pixels

        else:
            weight_mask = torch.ones_like(mask)
            if to_ignore is not None:
                for obj in to_ignore:
                    weight_mask[mask == obj] = 0

        return weight_mask


class CELossWeighted(WeightedLoss):
    def __init__(self, weighted=False):
        super(CELossWeighted, self).__init__()
        self.CrossEntropyLoss = nn.CrossEntropyLoss(reduction='none')
        self.weighted = weighted

    def forward(self, x, target):
        temp = self.CrossEntropyLoss(x, target)
        weight_mask = self.generate_weight_mask(target)
        loss = torch.sum(temp * weight_mask) / torch.sum(weight_mask)

        return loss


class CELossWeightedMasked(WeightedLoss):
    def __init__(self, weighted=False):
        super(CELossWeightedMasked, self).__init__()
        self.CrossEntropyLoss = nn.CrossEntropyLoss(reduction='none')
        self.weighted = weighted

    def forward(self, x, target, fg_mask):
        temp = self.CrossEntropyLoss(x, target)
        weight_mask = self.generate_weight_mask(fg_mask, to_ignore=[0])
        loss = torch.sum(temp * weight_mask) / torch.sum(weight_mask)

        return loss

def create_M_GT(foreground_labels):
    new_label = torch.zeros_like(foreground_labels)

    obj_index = 0
    for k in torch.unique(foreground_labels):

        if k in [0]:
            continue
        new_label[foreground_labels == k] = obj_index
        obj_index += 1

    return new_label.long()

class BCEWithLogitsLossWeighted(WeightedLoss):
    def __init__(self, weighted=False):
        super(BCEWithLogitsLossWeighted, self).__init__()
        self.BCEWithLogitsLoss = nn.BCEWithLogitsLoss(reduction='none')
        self.weighted = weighted

    def forward(self, x, target):
        temp = self.BCEWithLogitsLoss(x, target)
        weight_mask = self.generate_weight_mask(target)
        loss = torch.sum(temp * weight_mask) / torch.sum(weight_mask)

        return loss

class SmoothL1LossWeighted(WeightedLoss):
    def __init__(self, weighted=False):
        super(SmoothL1LossWeighted, self).__init__()
        self.SmoothL1Loss = nn.SmoothL1Loss(reduction='none')
        self.weighted = weighted

    def forward(self, x, target, mask=None):
        temp = self.SmoothL1Loss(x, target).sum(dim=1)
        if mask is None:
            return torch.sum(temp) / temp.numel()

        weight_mask = self.generate_weight_mask(mask)
        loss = torch.sum(temp * weight_mask) / torch.sum(weight_mask)

        return loss

class ClusterLossWeighted(WeightedLoss):
    def __init__(self, delta, weighted=False):
        super(ClusterLossWeighted, self).__init__()
        self.weighted=weighted
        self.delta = delta

    def forward(self, x1, y1, x2, y2):
        weight_vector_1 = self.generate_weight_mask(y1.unsqueeze(0))[0]
        weight_vector_2 = self.generate_weight_mask(y2.unsqueeze(0))[0]
        weight_matrix = torch.ger(weight_vector_1, weight_vector_2)
        indicator_matrix = (y1.unsqueeze(1) == y2.unsqueeze(0)).long()
        distance_matrix = euclidean_distances(x1,x2)
        positive_loss_matrix = indicator_matrix * distance_matrix**2
        negative_loss_matrix = (1 - indicator_matrix) * torch.clamp(self.delta - distance_matrix, min=0)**2
        return (weight_matrix * (positive_loss_matrix + negative_loss_matrix)).sum()