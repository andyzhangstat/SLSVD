
import numpy as np

def mse(X, Y):
    tmp = np.linalg.norm(np.dot(X, X.T) - np.dot(Y, Y.T), ord='fro')**2
    return tmp



def principle_angle(X, B):
    X = np.array(X)
    B = np.array(B)
    
    tmp = np.dot(X.T, B)
    _, _, Vh = np.linalg.svd(tmp)
    tmp1 = np.arccos(np.min(np.linalg.svd(tmp, compute_uv=False)))*(180/np.pi)
    
    return tmp1



def true_positive_rate(X, B):
    X = np.array(X)
    tmp = 0
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if B[i,j] != 0 and X[i,j] != 0:
                tmp += 1
    return tmp / np.count_nonzero(B)


def false_positive_rate(X, B):
    X = np.array(X)
    tmp = 0
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if B[i,j] == 0 and X[i,j] != 0:
                tmp += 1
    return tmp / (X.shape[0] * X.shape[1] - np.count_nonzero(B))


