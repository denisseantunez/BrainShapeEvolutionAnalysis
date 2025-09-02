import numpy as np
import torch
from scipy.sparse.linalg import eigs

def eigen_values_spectrum(laplace,ne):
            
        """
        Computes the spectral decomposition of the graph laplcian (eigen values and eigen vectors)

        inputs  : Graph Laplacian (Random walk/ normalized matrix)
        ne      : number of eigen values
 
        returns: Sorted eigen values and eigen vectors

        """
        eig_vals, eig_vecs = eigs(laplace,k = ne+1,sigma = 0, maxiter= 5000,tol = 1e-3)
        eig_vecs = np.real(eig_vecs[:, eig_vals.argsort()])

        signf = 1 - 2*(eig_vecs[0,:]<0)
        eig_vecs *= signf

        eig_vals = eig_vals.real[1:]
        eig_vecs = eig_vecs[:,1:]
        return torch.from_numpy(eig_vals), torch.from_numpy(eig_vecs)