import numpy as np
import scipy as sp
import cvxpy as cp
from qss import proximal


def polish(g, zk1, P, q, A, nuk1, dim):
    subdiff_center, subdiff_radius = proximal.get_subdiff(g, zk1)
    fixed_x_indices = (subdiff_radius != 0)
    num_fixed = np.sum(fixed_x_indices)
    fixed_r = np.zeros(dim)
    fixed_r[~fixed_x_indices] = subdiff_center[~fixed_x_indices]
    rhs = (
        -np.squeeze(np.asarray(P[:, fixed_x_indices].sum(axis=1)))
        - q
        - fixed_r
        - A.T @ nuk1
    )

    polish_A = sp.sparse.lil_matrix((dim, dim))
    ident = sp.eye(dim)
    polish_A[:, fixed_x_indices] = ident[:, fixed_x_indices]
    polish_A[:, ~fixed_x_indices] = P[:, ~fixed_x_indices]
    polish_A = sp.sparse.csc_matrix(polish_A)

    print("DIM:", dim)
    # print("RANK:", np.linalg.matrix_rank(polish_A.todense()))

    new_vecs = sp.sparse.linalg.lsqr(polish_A, rhs)[0]
    print(new_vecs)
    # TODO: ignore bad values of xu for indicator functions
    zk1[~fixed_x_indices] = new_vecs[~fixed_x_indices]

    return zk1