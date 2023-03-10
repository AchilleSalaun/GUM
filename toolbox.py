import numpy as np
import torch

from torch.linalg import det, inv
from torch        import eye, zeros, log, sqrt
from numpy        import pi

from torch.distributions import MultivariateNormal

# TENSORS MANAGEMENT
def tensor(l):
    return torch.tensor(l, dtype=torch.float32)

def identity(n):
    return eye(n)

def transpose(m):
    return torch.transpose(m, dim0=0, dim1=1)

def flatten(m, requires_grad=False):
    n = len(m)
    triangular = torch.linalg.cholesky(m) if n > 1 else m
    return tensor([
        triangular[i][j]
        for i in range(n)
        for j in range(i+1)
    ]).requires_grad_(requires_grad)

def unflatten(flattened_parameters):
    n = int((-1 + np.sqrt(1+8*len(flattened_parameters)))/2)
    t = zeros((n, n))
    for i in range(n):
        for j in range(i+1):
            n_ = i + j
            t[i][j] = torch.abs(flattened_parameters[n_]) if i == j else flattened_parameters[n_]
    return t @ transpose(t) if n > 1 else t

def detach(t):
    output = t.detach().numpy()
    if t.size() == (1, 1):
        return output[0, 0]
    return output

# STATISTICS
def rand(min_, max_):
    return (max_ - min_ ) * np.random.rand() + min_

def randn(mean, cov):
    if (cov == 0).all():
        return mean
    else:
        return transpose(MultivariateNormal(transpose(mean), cov).sample())

def log_gaussian(x, mu, sigma):
    k = len(sigma)
    return -log(sqrt((2 * pi) ** k * det(sigma))) - .5 * (transpose(x - mu) @ inv(sigma) @ (x - mu))
