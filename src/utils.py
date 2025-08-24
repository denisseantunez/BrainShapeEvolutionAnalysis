'''
    Adapted code from https://github.com/tiborkubik/toothForge 

    Functions responsible for aligning spectral embeddings with Iterative Closest Point (ICP)
    and k nearest neighbors.

'''

import math
import trimesh

import numpy as np

from scipy.spatial import cKDTree, KDTree
from scipy.sparse import coo_matrix

import SpectralMesh


def find_closest_points_in_spectral_domain(X_1, X_2):
    """
    Find nearest-neighbor correspondences between two point sets in spectral space.
    """
    tree_X_2 = cKDTree(X_2)
    _, corr_1_2 = tree_X_2.query(X_1)  # all corresponding points from embedding 1 onto embedding 2

    del tree_X_2

    return corr_1_2 # Indices of closest points in X_2 for each point in X_1.


def find_rotation_closed_form_iterative(m1: SpectralMesh,
                                        m2: SpectralMesh,
                                        opts: dict,
                                        data_term: bool = False,
                                        ) -> dict:
    """
    Align two meshes in spectral space using iterative closest point.

    Args:
        m1 (SpectralMesh) : First spectral mesh.
        m2 (SpectralMesh) : Second spectral mesh.
        opts (dict) :
            Dictionary of options containing:
                - 'niter': list[int]   number of iterations per reconstruction stage
                - 'kr': list[list[int]]  eigenmode indices used at each stage
        data_term (bool, optional) :
            If True, additional per-vertex data (m1.extended_data, m2.extended_data) is used
            in the embedding to guide alignment.

    Returns:
        corr (dict) :
            Dictionary with the following entries:
            - 'corr_12': nearest neighbor correspondences from m1 -> m2
            - 'corr_21': nearest neighbor correspondences from m2 -> m1
            - 'C_12': sparse correspondence matrix (m1->m2)
            - 'C_21': sparse correspondence matrix (m2->m1)
            - 'R_12': spectral rotation matrix (m1->m2)
            - 'R_21': spectral rotation matrix (m2->m1)
    """
    Z_1 = m1.graph.X[:, :3]
    Z_2 = m2.graph.X[:, :3]

    corr: dict = {
        'corr_12': None,
        'corr_21': None,
        'C_12': None,
        'C_21': None,
        'R_12': None,
        'R_21': None,
        
    }

    # In case you have other extra data per vertex
    if data_term:
        assert m1.extended_data is not None
        assert m2.extended_data is not None
        Z_1 = np.concatenate((Z_1, m1.extended_data), axis=1)
        Z_2 = np.concatenate((Z_2, m2.extended_data), axis=1)

    Z_1_o = Z_1
    Z_2_o = Z_2

    last_err_Z = 1e10
    last_Z_1 = Z_1
    last_Z_2 = Z_2

    errs_X = list()
    errs_Z = list()
    print(f'Align Embeddings')


    for iter_recon in range(len(opts['niter'])):
        # Perform the inner loop using the value in the list as the number of iterations
        for iter in range(opts['niter'][iter_recon]):
            corr['corr_12'] = find_closest_points_in_spectral_domain(Z_1_o,
                                                                     Z_2)  # closest points of M1 in M2. Shape: [n_z_1, ]
        
            corr['corr_21'] = find_closest_points_in_spectral_domain(Z_2_o, Z_1) # closest points of M2 in M1.
          
            # Create correspondence matrices for the two cases
            C_12 = coo_matrix((np.ones(Z_1_o.shape[0]),
                               (np.arange(Z_1_o.shape[0]), corr['corr_12'])),
                              shape=(Z_1_o.shape[0], Z_2_o.shape[0]))
            C_21 = coo_matrix((np.ones(Z_2_o.shape[0]),
                               (np.arange(Z_2_o.shape[0]), corr['corr_21'])),
                              shape=(Z_2_o.shape[0], Z_1_o.shape[0]))

            # Compute errors
            err_X = np.sum(np.sum((Z_1[corr['corr_21'], :3] - Z_2_o[:, :3]) ** 2, axis=1))
            err_Z = np.sum(np.sum((Z_1[corr['corr_21'], :] - Z_2_o) ** 2, axis=1))
            errs_X.append(err_X)
            errs_Z.append(err_Z)
            print(f'[{iter_recon} - {iter}/{opts["niter"][iter_recon]}, {opts["kr"][iter_recon][-1] + 1} eigenmodes] '
                  f'Total sum of squared differences X: {err_X}'
                  f'Total sum of squared differences Z: {err_Z}')

            # Stop if error increases
            if err_Z > last_err_Z:
                Z_1 = last_Z_1
                Z_2 = last_Z_2
                break

            last_err_Z = err_Z
            last_Z_1 = Z_1
            last_Z_2 = Z_2

            # Direction 1
            R_21 = m1.graph.eig_vecs_inv[opts['kr'][iter_recon], :] @ C_12 @ m2.graph.eig_vecs_inv[
                                                                             opts['kr'][iter_recon], :].T
            w_1 = m1.graph.eig_vecs_inv[opts['kr'][iter_recon], :] @ m1.graph.X[:, opts['kr'][iter_recon]]
            w_2 = m2.graph.eig_vecs_inv[opts['kr'][iter_recon], :] @ m2.graph.X[:, opts['kr'][iter_recon]]
            w = R_21 @ w_2

            Y_1 = m1.graph.eig_vecs[:, opts['kr'][iter_recon]] @ w
            Y_1 = Y_1[:, :3]

            if data_term:
                Z_1 = Y_1
                Z_1 = np.concatenate((Z_1, m1.extended_data), axis=1)
            else:
                Z_1 = Y_1

            # Direction 2
            R_12 = m2.graph.eig_vecs_inv[opts['kr'][iter_recon], :] @ C_21 @ m1.graph.eig_vecs_inv[
                                                                             opts['kr'][iter_recon], :].T
            w = R_12 @ w_1
            Y_2 = m2.graph.eig_vecs[:, opts['kr'][iter_recon]] @ w

            Y_2 = Y_2[:, :3]

            if data_term:
                Z_2 = Y_2
                Z_2 = np.concatenate((Z_2, m2.extended_data), axis=1)
            else:
                Z_2 = Y_2

    # Save last correspondences and transforms
    corr['C_12'] = C_12
    corr['C_21'] = C_21
    corr['R_12'] = R_12
    corr['R_21'] = R_21


    return corr

