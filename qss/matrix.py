import numpy as np
import scipy as sp


def build_kkt(P, A, reg, rho, dim, constr_dim):
    return sp.sparse.vstack(
        [
            sp.sparse.hstack([P + rho * sp.sparse.identity(dim), A.T]),
            sp.sparse.hstack([A, reg * sp.sparse.eye(constr_dim)]),
        ]
    )


def ir_solve(A, Atilde, b):
    tol = 1e-7
    lk = Atilde.solve(b)
    k = 0
    while np.linalg.norm(A @ lk - b, ord=np.inf) > tol:
        lk = lk + Atilde.solve(b - A @ lk)
        k += 1
    """
    if k != 0:
        print("Did", k, "rounds of iterative refinement")
    """
    return lk
