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




def sparse_logistic_svd_coord(dat, lambdas=np.logspace(-2, 2, num=10), k=2, quiet=True,
                               max_iters=100, conv_crit=1e-5, randstart=False,
                               normalize=False, start_A=None, start_B=None, start_mu=None):
    """
    Sparse Logistic SVD with Coordinate Descent.

    Parameters:
    dat : ndarray
        Input data matrix.
    lambdas : array_like, optional
        Array of regularization parameters.
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
        Tuple containing mu, A, B, zeros_mat, BICs.
        - mu: The mean vector.
        - A: The matrix A.
        - B: The matrix B.
        - zeros_mat: Matrix indicating the number of zeros in each component.
        - BICs: Matrix containing the Bayesian Information Criterion for each component.
    """


    q = 2 * dat - 1
    q[np.isnan(q)] = 0

    n, d = dat.shape

    if not randstart:
        mu = np.mean(q, axis=0)
        udv = svd((q - np.mean(q, axis=0)).T, full_matrices=False)
        # A = udv[0][:, :k]
        # B = udv[2][:k, :].T @ np.diag(udv[1][:k])
        B = udv[0][:, :k]
        #A = udv[2][:k, :].T @ np.diag(udv[1][:k])
        A = udv[2][:k, :].T
    else:
        mu = np.random.randn(d)
        A = np.random.uniform(-1, 1, size=(n, k))
        B = np.random.uniform(-1, 1, size=(d, k))

    if start_B is not None:
        #B = start_B * np.sqrt(np.sum(start_A**2, axis=0))
        B = start_B / np.sqrt(np.sum(start_B**2, axis=0))

    if start_A is not None:
        A = start_A / np.sqrt(np.sum(start_A**2, axis=0))

    if start_mu is not None:
        mu = start_mu

    BICs = np.zeros((len(lambdas), k))
    zeros_mat = np.zeros((len(lambdas), k))
    iters = np.zeros((len(lambdas), k))

    theta = np.outer(np.ones(n), mu) + (A @ B.T)
    X = theta + 4 * q * (1 - inv_logit_mat(q * theta))
    Xcross = X - (A @ B.T)
    mu = np.mean(Xcross, axis=0)

    for m in range(k):
        last_A = A.copy()
        last_B = B.copy()

        theta = np.outer(np.ones(n), mu) + (A @ B.T)
        X = theta + 4 * q * (1 - inv_logit_mat(q * theta))
        Xm = X - (np.outer(np.ones(n), mu) + A[:, np.arange(k) != m] @ B[:, np.arange(k) != m].T)

        Bms = np.zeros((d, len(lambdas)))
        Ams = np.zeros((n, len(lambdas)))

        for j, lambda_val in enumerate(lambdas):
            for i in range(max_iters):
                if np.sum(B[:, m]**2) == 0:
                    A[:, m] = Xm @ B[:, m]
                    break

                A[:, m] = Xm @ B[:, m] / np.sum(B[:, m]**2)
                A[:, m] = A[:, m] / np.sqrt(np.sum(A[:, m]**2))

                B_lse = Xm.T @ A[:, m]
                B[:, m] = np.sign(B_lse) * np.maximum(0, np.abs(B_lse) - lambda_val)

                loglike = np.sum(np.log(inv_logit_mat(q * (np.outer(np.ones(n), mu) + (A @ B.T))))[~np.isnan(dat)])
                penalty = 0.25 * lambda_val * np.sum(np.abs(B[:, m]))
                cur_loss = (-loglike + penalty) / np.sum(~np.isnan(dat))

                if not quiet:
                    print(m, "  ", np.round(-loglike, 4), "   ", np.round(penalty, 4),
                          "     ", np.round(-loglike + penalty, 4))

                if i > 4:
                    if (last_loss - cur_loss) / last_loss < conv_crit:
                        break

                last_loss = cur_loss

            Bms[:, j] = B[:, m] / (1 if np.sum(B[:, m]**2) == 0 else np.sqrt(np.sum(B[:, m]**2)))
            Ams[:, j] = Xm @ Bms[:, j] / (1 if np.sum(Bms[:, j]**2) == 0 else np.sum(Bms[:, j]**2))

            #BICs[j, m] = -2 * loglike + np.log(n * d) * (np.sum(np.abs(B) >= 1e-10))
            BICs[j, m] = -2 * loglike + np.log(n * d) * (np.count_nonzero(B[:, m]))


            #zeros_mat[j, m] = np.sum(np.abs(B[:, m]) < 1e-10)
            zeros_mat[j, m] = np.count_nonzero(B[:, m])

            iters[j, m] = i

        B[:, m] = Bms[:, np.argmin(BICs[:, m])]
        A[:, m] = Ams[:, np.argmin(BICs[:, m])]

    if normalize:
        #A = A / np.sqrt(np.sum(B**2, axis=0))
        A = A / np.sqrt(np.sum(A**2, axis=0))
        B = B / np.sqrt(np.sum(B**2, axis=0))


    return mu, A, B, zeros_mat, BICs

