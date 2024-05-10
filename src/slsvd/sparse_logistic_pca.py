


import numpy as np
from numpy.linalg import svd

def inv_logit_mat(x, min_val=0, max_val=1):
    p = np.exp(x) / (1 + np.exp(x))
    which_large = np.isnan(p) & ~np.isnan(x)
    p[which_large] = 1
    return p * (max_val - min_val) + min_val

def sparse_logistic_pca(dat, lambda_val=0, k=2, quiet=True, max_iters=100, conv_crit=1e-5,
                        randstart=False, procrustes=True, lasso=True, normalize=False,
                        start_A=None, start_B=None, start_mu=None):
    q = 2 * dat - 1
    q[np.isnan(q)] = 0

    n, d = dat.shape

    if not randstart:
        mu = np.mean(q, axis=0)
        udv = svd((q - np.mean(q, axis=0)).T, full_matrices=False)
        # A = udv[0][:, :k]
        # B = udv[2][:k, :].T @ np.diag(udv[1][:k])
        A = udv[2][:k, :].T @ np.diag(udv[1][:k])
        B = udv[0][:, :k]
        

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
            # A = Xstar @ B @ np.linalg.inv(B.T @ B)
            # A, _ = np.linalg.qr(A)
            A = Xstar @ B @ np.linalg.inv(B.T @ B)
            A, _ = np.linalg.qr(A)

        theta = np.outer(np.ones(n), mu) + (A @ B.T)
        X = theta + 4 * q * (1 - inv_logit_mat(q * theta))
        Xstar = X - np.outer(np.ones(n), mu)

        if lasso:
            B_lse = Xstar.T @ A
            B = np.sign(B_lse) * np.maximum(0, np.abs(B_lse) - 4 * n * lambda_val)
        else:
            # C = Xstar.T @ A
            # B = np.abs(B) / (np.abs(B) + 4 * n * lambda_val) * C
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
    BIC = -2 * loglike + np.log(n) * (d + n * k + np.sum(np.abs(B) >= 1e-10))

    # return {'mu': mu, 'A': A, 'B': B, 'zeros': zeros, 'BIC': BIC, 'iters': m,
    #         'loss_trace': loss_trace[:m + 1], 'lambda': lambda_val}
    return mu, A, B, zeros, BIC, m, loss_trace[:m + 1], lambda_val





def sparse_logistic_pca_coord(dat, lambdas=np.logspace(-2, 2, num=10), k=2, quiet=True,
                               max_iters=100, conv_crit=1e-5, randstart=False,
                               normalize=False, start_A=None, start_B=None, start_mu=None):
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

    #zeros = np.sum(np.abs(B) < 1e-10)
    #BIC = -2 * loglike + np.log(n * d) * (np.sum(np.abs(B) >= 1e-10))
    #BIC = -2 * loglike + np.log(n * d) * (np.sum(np.abs(B) >= 1e-10))



    # return {'mu': mu, 'A': A, 'B': B, 'zeros': zeros, 'zeros_mat': zeros_mat,
    #         'BICs': BICs, 'BIC': BIC, 'lambdas': lambdas, 'iters': iters}
#loss_trace[:m + 1],

    return mu, A, B, zeros_mat, BICs, m,  lambda_val
