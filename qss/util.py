import numpy as np
import scipy as sp


def evaluate_stop_crit(xk1, zk, zk1, uk1, dim, rho, eps_abs, eps_rel):
    epri = np.sqrt(dim) * eps_abs + eps_rel * max(
        np.linalg.norm(xk1, ord=np.inf), np.linalg.norm(zk1, ord=np.inf)
    )
    edual = np.sqrt(dim) * eps_abs + eps_rel * np.linalg.norm(rho * uk1, ord=np.inf)
    if (
        np.linalg.norm(xk1 - zk1, ord=np.inf) < epri
        and np.linalg.norm(rho * (zk - zk1), ord=np.inf) < edual
    ):
        return True

    return False
