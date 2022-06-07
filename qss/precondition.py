import numpy as np
import scipy as sp


def ruiz(P, q, r, A, b):
    dim = P.shape[0]
    constr_dim = A.shape[0]
    eps_equil = 1e-4

    c = 1
    S1 = sp.sparse.identity(dim)
    S2 = sp.sparse.identity(constr_dim)
    delta1 = np.zeros(dim)
    delta2 = np.zeros(constr_dim)
    Pbar = P.copy()
    qbar = np.copy(q)
    Abar = A.copy()
    bbar = np.copy(b)
    rbar = r

    while (
        np.linalg.norm(1 - delta1, ord=np.inf) > eps_equil
        and np.linalg.norm(1 - delta2, ord=np.inf) > eps_equil
    ):
        Pbarnorm = np.sqrt(sp.sparse.linalg.norm(Pbar, ord=np.inf, axis=0))
        Abarnorm = np.sqrt(sp.sparse.linalg.norm(Abar, ord=np.inf, axis=0))
        ATbarnorm = np.sqrt(sp.sparse.linalg.norm(Abar, ord=np.inf, axis=1))

        Pbarnorm[Pbarnorm == 0] = 1
        Abarnorm[Abarnorm == 0] = 1
        ATbarnorm[ATbarnorm == 0] = 1

        delta1 = 1 / np.maximum(Pbarnorm, Abarnorm)
        delta2 = 1 / ATbarnorm

        Delta1 = sp.sparse.diags(delta1)
        Delta2 = sp.sparse.diags(delta2)

        Pbar = Delta1 @ Pbar @ Delta1
        Abar = Delta2 @ Abar @ Delta1
        qbar = Delta1 @ qbar
        bbar = Delta2 @ bbar

        gamma = 1 / max(
            # TODO: this is redundant - will be calculated in the next iter too
            np.mean(sp.sparse.linalg.norm(Pbar, ord=np.inf, axis=0)),
            np.linalg.norm(qbar, ord=np.inf),
        )

        Pbar *= gamma
        qbar *= gamma
        rbar *= gamma

        S1 = Delta1 @ S1
        S2 = Delta2 @ S2

        c *= gamma

    return (Pbar, qbar, rbar, Abar, bbar, S1.diagonal(), c)
