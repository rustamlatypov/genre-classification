# import the needed libraries
import matplotlib.pyplot as plt
import numpy as np


def response_matrix(y):
    # Input: label vector y of length N
    # Output: N by l matrix Y of respones variables
    N = y.shape[0]
    classes = np.unique(y)
    # amount of classes l
    l = classes.shape[0]
    Y = np.ones((N, l))
    for i in classes:
        ind = np.where(classes == i)[0][0]
        Y[:, ind] = (y == i).astype(int)
    return Y


def compute_spca(X, y, d):
    # Input: the N by D data matrix Z, the N by l response variable matrix Y, the number of components d
    # Output: a d by D matrix W_pca, and all eigenvalues of Q
    Y = response_matrix(y)

    # step1: compute the sample cov. matrix Q
    Q = np.dot(np.dot(np.dot(np.transpose(X), Y), np.transpose(Y)), X)

    # step2: compute the eigenvalues and eigenvectors
    u, s, vh = np.linalg.svd(Q)

    # step3: Sort the eigenvectors by decreasing eigenvalues, choose the d largest eigenvalues, form W_pca
    idx = s.argsort()[::-1]
    eigenValues = s[idx]
    eigenVectors = u[:, idx]

    eigvalues = eigenValues[0:d]
    W_pca = eigenVectors[:, 0:d]

    return W_pca, eigvalues


def plot_error(eigvalues, max_d):
    x = range(1, max_d + 1)
    errors = [sum(eigvalues[d:]) for d in x]
    plt.plot(x, errors)
    plt.xlabel('Number of principal components $d$')
    plt.ylabel('Reconstruction error $\mathcal{E}$')
    plt.title('Number of principal components vs the reconstruction error')
    plt.show()


def plot_scores(T, i, j):
    # i and j are the principal components we are plotting
    plt.figure(figsize=(8, 8))
    p1 = plt.scatter(T[:, i], T[:, j])
    plt.show(p1)

