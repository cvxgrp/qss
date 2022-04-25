import numpy as np
import scipy as sp
import qdldl
from qss import precondition
from qss import matrix
from qss import proximal
from qss import util


class QSS(object):
    def __init__(
        self,
        data,
        eps_abs=1e-4,
        eps_rel=1e-4,
        alpha=1.4,
        rho=0.1,
        precond=True,
        reg=True,
        use_iter_refinement=True,
    ):
        self._data = data
        self._eps_abs = eps_abs
        self._eps_rel = eps_rel
        self._alpha = alpha
        self._rho = rho
        self._precond = precond
        self._reg = reg
        self._use_iter_refinement = use_iter_refinement
        return

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
        has_constr = True if A.nnz != 0 else False
        if not has_constr:
            constr_dim = 0

        xk = np.zeros(dim)
        zk = np.zeros(dim)
        uk = np.zeros(dim)

        xk1 = np.zeros(dim)
        zk1 = np.zeros(dim)
        uk1 = np.zeros(dim)

        equil_scaling = np.ones(dim)
        c = 1

        # Scaling
        if self._precond:
            # We are now solving for xtilde, where x = equil_scaling * xtilde
            P, q, r, A, b, equil_scaling, c = precondition.ruiz(P, q, r, A, b)

        # Constructing KKT matrix
        if has_constr:
            quad_kkt = matrix.build_kkt(P, A, 0, rho, dim, constr_dim)
            quad_kkt_reg = matrix.build_kkt(P, A, -1e-7, rho, dim, constr_dim)
            F = qdldl.Solver(quad_kkt_reg)
        else:
            quad_kkt = P + rho * sp.sparse.identity(dim)
            F = qdldl.Solver(quad_kkt)

        i = 0
        while True:
            i += 1

            # Update x
            if has_constr:
                if self._use_iter_refinement:
                    xk1 = matrix.ir_solve(
                        quad_kkt, F, np.concatenate([-q + rho * (zk - uk), b])
                    )[:dim]
                else:
                    xk1 = F.solve(np.concatenate([-q + rho * (zk - uk), b]))[:dim]
            else:
                xk1 = F.solve(-q + rho * (zk - uk))

            # Update z
            zk1 = proximal.apply_prox_ops(
                rho / c, equil_scaling, g, alpha * xk1 + (1 - alpha) * zk + uk
            )

            # Update u
            uk1 = uk + alpha * xk1 + (1 - alpha) * zk - zk1

            # Calculate residuals
            r_prim = xk1 - zk1
            r_dual = rho * (zk - zk1)

            # Check if we should stop
            if i % 10 == 0 and util.evaluate_stop_crit(
                xk1, zk, zk1, uk1, dim, rho, self._eps_abs, self._eps_rel
            ):
                print("Finished in", i, "iterations")
                return (
                    (0.5 * zk1 @ P @ zk1 + q @ zk1 + r) / c
                    + proximal.apply_g_funcs(g, equil_scaling * zk1),
                    equil_scaling * zk1,
                )

            # Update rho
            if i % 10 == 0:
                new_rho_candidate = rho * np.sqrt(
                    np.linalg.norm(r_prim, ord=np.inf)
                    / np.linalg.norm(r_dual, ord=np.inf)
                    * np.linalg.norm(rho * uk1)
                    / max(
                        np.linalg.norm(xk1, ord=np.inf), np.linalg.norm(zk1, ord=np.inf)
                    )
                )

                # This is for the first iteration
                if new_rho_candidate == 0:
                    new_rho_candidate = rho

                if new_rho_candidate / rho > 5 or rho / new_rho_candidate > 5:
                    # print("CHANGING RHO from", rho, "TO", new_rho_candidate)
                    uk1 = uk1 * rho / new_rho_candidate

                    # Update KKT matrix
                    rho_vec = sp.sparse.diags(
                        np.concatenate([rho * np.ones(dim), np.zeros(constr_dim)])
                    )
                    new_rho_vec = sp.sparse.diags(
                        np.concatenate(
                            [new_rho_candidate * np.ones(dim), np.zeros(constr_dim)]
                        )
                    )
                    quad_kkt = quad_kkt - rho_vec + new_rho_vec
                    if has_constr:
                        quad_kkt_reg = quad_kkt_reg - rho_vec + new_rho_vec
                        F.update(quad_kkt_reg)
                    else:
                        F.update(quad_kkt)

                    rho = new_rho_candidate

            xk = xk1
            zk = zk1
            uk = uk1
