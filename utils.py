import torch as t
import numpy as np


def get_sparsity_score(vec):
    """
    Get Sparsity Score
    
    Computes an normalized sparsity score of a vector of proportions
    that sum to one.

    :param vec: Tensor - one dimensional
    :returns: Float - Normalized score
    """
    K = np.prod(np.array(vec.size()))
    uniform_vec = t.tensor([1/K for _ in range(K)]).float()
    max_sparsity = 2 * ((K - 1) / K) 
    norm_score = sum(abs(vec.float().to('cpu') - uniform_vec.float())) / max_sparsity
    
    return norm_score.item()

