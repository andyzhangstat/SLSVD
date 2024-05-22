import numpy as np
from scipy.special import expit

def generate_data(n, d, rank, random_seed=123):
    """Generate binary data matrix.

    Parameters
    ----------
    n : integer
        The number of data points.
    d : integer
        The number of features.
    rank : integer
        The number of rank.
    random_seed : integer
        Random seed to ensure reproducibility.

    Returns
    -------
    X : ndarray
        Binary data matrix of shape (n, d).


    Examples
    --------
    >>> from slsvd.data_generation import generate_data
    >>> generate_data_lr(n=50, d=100, rank=2, random_seed=123)
    """
    
    if not isinstance(n, int):
        raise ValueError('Sample size n must be an integer')

    if not isinstance(d, int):
        raise ValueError('Number of features d must be an integer')

    if not isinstance(rank, int):
        raise ValueError('Rank must be an integer')

    np.random.seed(random_seed)
    
    # Construct a low-rank matrix in the logit scale
    loadings = np.zeros((d, rank))
    loadings[:20, 0] = 1
    loadings[20:40, 1] = 1
    
    def gram_schmidt(matrix):
        q, r = np.linalg.qr(matrix)
        return q

    loadings = gram_schmidt(loadings)
    scores = np.random.normal(size=(n, rank))
    diagonal = np.diag((10, 5))

    mat_logit = np.dot(scores, np.dot(loadings, diagonal).T)

    # Compute the inverse of the logit function
    inverse_logit_mat = expit(mat_logit)

    bin_mat = np.random.binomial(1, inverse_logit_mat)

    return bin_mat, loadings, scores, diagonal

