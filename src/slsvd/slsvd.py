import numpy as np
from numpy.linalg import svd

def inv_logit_mat(x, min_val=0, max_val=1):
    """Inverse logit transformation.

    Parameters:
    x : ndarray
        Input array.
    min_val : float, optional
        Minimum value for the output range.
    max_val : float, optional
        Maximum value for the output range.

    Returns:
    ndarray
        Inverse logit transformed array.
    """
    p = np.exp(x) / (1 + np.exp(x))
    which_large = np.isnan(p) & ~np.isnan(x)
    p[which_large] = 1
    return p * (max_val - min_val) + min_val

def sparse_logistic_pca(dat, lambda_val=0, k=2, quiet=True, max_iters=100, conv_crit=1e-5,
                        randstart=False, procrustes=True, lasso=True, normalize=False,
                        start_A=None, start_B=None, start_mu=None):
    """Sparse Logistic PCA.

    Parameters:
    dat : ndarray
        Input data matrix.
    lambda_val : float, optional
        Regularization parameter.
    k : int, optional
        Number of components.
    quiet : bool, optional
        If True, suppresses iteration printouts.
    max_iters : int, optional
        Maximum number of iterations.
    conv_crit : float, optional
        Convergence criterion.
    randstart : bool, optional
        If True, uses random initialization.
    procrustes : bool, optional
        If True, uses procrustes transformation.
    lasso : bool, optional
        If True, applies lasso regularization.
    normalize : bool, optional
        If True, normalizes the components.
    start_A : ndarray, optional
        Initial value for matrix A.
    start_B : ndarray, optional
        Initial value for matrix B.
    start_mu : ndarray, optional
        Initial value for mean vector.

    Returns:
    tuple
        Tuple containing mu, A, B, zeros, BIC, m, lambda_val.
    """
    q = 2 * dat - 1
    q[np.isnan(q)] = 0

    n, d = dat.shape

    if not randstart:
        mu = np.mean(q, axis=0)
        udv = svd((q - np.mean(q, axis=0)).T, full_matrices=False)
        B = udv[0][:, :k]
        A = udv[2][:k, :].T @ np.diag(udv[1][:k])
    else:
        mu = np.random.randn(d)
        A = np.random.uniform(-1, 1, size=(n, k))
        B = np.random.uniform(-1, 1, size=(d, k))

    if start_B is not None:
        B = start_B * np.sqrt(np.sum(start_A**2, axis=0))

    if start_A is not None:
        A = start_A / np.sqrt(np.sum(start_A**2, axis=0))

    if start_mu is not None:
        mu = start_mu

    loss_trace = np.zeros(max_iters)

    for m in range(max_iters):
        last_mu = mu
        last_A = A.copy()
        last_B = B.copy()

        theta = np.outer(np.ones(n), mu) + (A @ B.T)
        X = theta + 4 * q * (1 - inv_logit_mat(q * theta))
        Xcross = X - (A @ B.T)
        mu = np.mean(Xcross, axis=0)

        theta = np.outer(np.ones(n), mu) + (A @ B.T)
        X = theta + 4 * q * (1 - inv_logit_mat(q * theta))
        Xstar = X - np.outer(np.ones(n), mu)

        if procrustes:
            M = svd(Xstar @ B)
            A = M[0][:,:2] @ M[2].T
        else:
            A = Xstar @ B @ np.linalg.inv(B.T @ B)
            A, _ = np.linalg.qr(A)

        theta = np.outer(np.ones(n), mu) + (A @ B.T)
        X = theta + 4 * q * (1 - inv_logit_mat(q * theta))
        Xstar = X - np.outer(np.ones(n), mu)

        if lasso:
            B_lse = Xstar.T @ A
            B = np.sign(B_lse) * np.maximum(0, np.abs(B_lse) - 4 * n * lambda_val)
        else:
            C = Xstar.T @ A
            B = np.abs(B) / (np.abs(B) + 4 * n * lambda_val) * C

        loglike = np.sum(np.log(inv_logit_mat(q * (np.outer(np.ones(n), mu) + (A @ B.T))))[~np.isnan(dat)])
        penalty = n * lambda_val * np.sum(np.abs(B))
        loss_trace[m] = (-loglike + penalty) / np.sum(~np.isnan(dat))

        if not quiet:
            print(m, "  ", np.round(-loglike, 4), "   ", np.round(penalty, 4), "     ", np.round(-loglike + penalty, 4))

        if m > 4:
            if loss_trace[m - 1] - loss_trace[m] < conv_crit:
                break

    if loss_trace[m - 1] < loss_trace[m]:
        mu = last_mu
        A = last_A
        B = last_B
        m = m - 1

        loglike = np.sum(np.log(inv_logit_mat(q * (np.outer(np.ones(n), mu) + (A @ B.T))))[~np.isnan(dat)])

    if normalize:
        A = A / np.sqrt(np.sum(B**2, axis=0))
        B = B / np.sqrt(np.sum(B**2, axis=0))

    zeros = np.sum(np.abs(B) < 1e-10)
    BIC = -2 * loglike + np.log(n * d) * (np.sum(np.abs(B) >= 1e-10))

    return mu, A, B, zeros, BIC, m, lambda_val