import numpy as np
import scipy as sp


def ruiz(data, scaling):
    dim = data["dim"]
    constr_dim = data["constr_dim"]
    eps_equil = 1e-4

    c = 1
    S1 = sp.sparse.identity(dim)
    S2 = sp.sparse.identity(constr_dim)
    delta1 = np.zeros(dim)
    delta2 = np.zeros(constr_dim)
    Pbar = data["P"]
    qbar = data["q"]
    Abar = data["A"]
    bbar = data["b"]
    rbar = data["r"]

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

        # TODO: Look into whether this is beneficial or not
        gamma = 1 / max(
            # TODO: this is redundant - will be calculated in the next iter too
            np.mean(sp.sparse.linalg.norm(Pbar, ord=np.inf, axis=0)),
            np.linalg.norm(qbar, ord=np.inf),
        )
        # gamma = 1

        Pbar *= gamma
        qbar *= gamma
        rbar *= gamma

        S1 = Delta1 @ S1
        S2 = Delta2 @ S2

        c *= gamma

    data["P"] = Pbar
    data["q"] = qbar
    data["r"] = rbar
    data["A"] = Abar
    data["b"] = bbar

    scaling["equil_scaling"] = S1.diagonal()
    scaling["obj_scale"] = c
