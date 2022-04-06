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

    def evaluate_stop_crit(self, xk, xk1, zk, zk1, uk, uk1, dim, rho):
        epri = np.sqrt(dim) * self._eps_abs + self._eps_rel * max(
            np.linalg.norm(xk1), np.linalg.norm(zk1)
        )
        edual = np.sqrt(dim) * self._eps_abs + self._eps_rel * np.linalg.norm(rho * uk1)
        if (
            np.linalg.norm(xk1 - zk1) < epri
            and np.linalg.norm(rho * (zk - zk1)) < edual
        ):
            return True

        return False

    def ruiz_equilibration(self, P, q, r, A, b):
        dim = P.shape[0]
        constr_dim = A.shape[0]
        eps_equil = 1e-4

        if A.nnz != 0:
            M = sp.sparse.vstack(
                [
                    sp.sparse.hstack([P, A.T]),
                    sp.sparse.hstack(
                        [A, sp.sparse.csc_matrix((constr_dim, constr_dim))]
                    ),
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
            norm_sqrt[norm_sqrt==0] = 1 # do this to prevent divide by zero
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

        rho_scaling = np.ones(dim)

        # Scaling
        P, q, r, A, b, rho_scaling = self.ruiz_equilibration(P, q, r, A, b)

        # Constructing KKT matrix
        if A.nnz != 0:
            quad_kkt = sp.sparse.vstack(
                [
                    sp.sparse.hstack([P + rho * sp.sparse.identity(dim), A.T]),
                    sp.sparse.hstack(
                        [A, sp.sparse.csc_matrix((constr_dim, constr_dim))]
                    ),
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
            zk1 = util.apply_prox_ops(
                rho / rho_scaling, g, alpha * xk1 + (1 - alpha) * zk + uk
            )

            # Update u
            uk1 = uk + alpha * xk1 + (1 - alpha) * zk - zk1

            if i % 100 == 0 and self.evaluate_stop_crit(
                xk, xk1, zk, zk1, uk, uk1, dim, rho
            ):
                print("Finished in", i, "iterations")
                return (
                    0.5 * zk1 @ P @ zk1
                    + q @ zk1
                    + r
                    + util.apply_g_funcs(g, rho_scaling * zk1),
                    rho_scaling * zk1,
                )

            xk = xk1
            zk = zk1
            uk = uk1
