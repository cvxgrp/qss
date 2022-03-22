import numpy as np
import scipy as sp
import qdldl
import util


class QSS(object):
    def __init__(self, data, eps_abs=1e-4, eps_rel=1e-4, alpha=1.4, rho=0.04):
        self._data = data
        self._eps_abs = eps_abs
        self._eps_rel = eps_rel
        self._alpha = alpha
        self._rho = rho
        return

    def evaluate_stop_crit(self, xk, zk, zk1, uk, dim, rho):
        epri = np.sqrt(dim) * self._eps_abs + self._eps_rel * max(
            np.linalg.norm(xk), np.linalg.norm(zk)
        )
        edual = np.sqrt(dim) * self._eps_abs + self._eps_rel * np.linalg.norm(rho * uk)
        if np.linalg.norm(xk - zk) < epri and np.linalg.norm(rho * (zk - zk1)) < edual:
            return True

        return False

    def solve(self):
        P = self._data["P"]
        q = self._data["q"]
        r = self._data["r"]
        A = self._data["A"]
        b = self._data["b"]
        g = self._data["g"]

        eps_abs = self._eps_abs
        eps_rel = self._eps_rel
        alpha = self._alpha
        rho = self._rho

        dim = P.shape[0]
        constr_dim = A.shape[0]

        xk = np.zeros(dim)
        zk = np.zeros(dim)
        uk = np.zeros(dim)

        xk1 = np.zeros(dim)
        zk1 = np.zeros(dim)
        uk1 = np.zeros(dim)

        # Scaling
        D = 2 * sp.sparse.identity(dim)
        E = 2 * sp.sparse.identity(constr_dim)
        P = D @ P @ D
        q = D @ q
        A = E @ A @ D
        b = E @ b
        rho_scaling = D.diagonal()

        # Constructing KKT matrix
        if A.nnz != 0:
            quad_kkt = sp.sparse.vstack(
                [
                    sp.sparse.hstack([P + rho * sp.sparse.identity(dim), A.T]),
                    sp.sparse.hstack([A, np.zeros((constr_dim, constr_dim))]),
                ]
            )
        else:
            quad_kkt = P + rho * sp.sparse.identity(dim)
        F = qdldl.Solver(quad_kkt)

        i = 0
        while True:
            i += 1

            # Update x
            if A.nnz != 0:
                xk1 = F.solve(np.concatenate([-q + rho * (zk - uk), b]))[:dim]
            else:
                xk1 = F.solve(-q + rho * (zk - uk))

            # Update z
            zk1 = util.apply_prox_ops(rho * rho_scaling, g, alpha * xk1 + (1 - alpha) * zk + uk)

            # Update u
            uk1 = uk + alpha * xk1 + (1 - alpha) * zk - zk1

            if i % 1 == 0 and self.evaluate_stop_crit(xk, zk, zk1, uk, dim, rho):
                print("Finished in", i, "iterations")
                # NOTE: If we also want x, need to recover with x = D @ xk1
                return 0.5 * xk1 @ P @ xk1 + q @ xk1 + r + util.apply_g_funcs(g, zk1)

            xk = xk1
            zk = zk1
            uk = uk1
