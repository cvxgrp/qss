import numpy as np
import scipy as sp


def ruiz(P, q, r, A, b):
    dim = P.shape[0]
    constr_dim = A.shape[0]
    eps_equil = 1e-4

    if A.nnz != 0:
        M = sp.sparse.vstack(
            [
                sp.sparse.hstack([P, A.T]),
                sp.sparse.hstack([A, sp.sparse.csc_matrix((constr_dim, constr_dim))]),
            ]
        )
        qb = np.concatenate([q, b])
    else:
        constr_dim = 0
        M = P
        qb = q

    c = 1
    S = sp.sparse.identity(dim + constr_dim)
    delta = np.zeros(dim + constr_dim)

    while np.linalg.norm(1 - delta, ord=np.inf) > eps_equil:
        norm_sqrt = np.sqrt(sp.sparse.linalg.norm(M, ord=np.inf, axis=0))
        norm_sqrt[norm_sqrt == 0] = 1  # do this to prevent divide by zero
        delta = 1 / norm_sqrt
        Delta = sp.sparse.diags(delta)
        M = Delta @ M @ Delta
        qb = Delta @ qb
        S = sp.sparse.diags(delta) @ S

    if A.nnz != 0:
        return (
            M[:dim, :dim],
            qb[:dim],
            r,
            M[dim:, :dim],
            qb[dim:],
            S.diagonal()[:dim],
        )
    else:
        return M, qb, r, A, b, S.diagonal()
