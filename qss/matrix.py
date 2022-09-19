import numpy as np
import scipy as sp
import qdldl
from qss import linearoperator


class KKT:
    def __init__(self, P, A, rho_controller):
        self._dim = A.shape[1]
        self._constr_dim = A.shape[0]
        self._has_constr = A.nnz != 0
        self._rho_controller = rho_controller
        self._rho_vec = np.copy(rho_controller.get_rho_vec())

        reg = -1e-7
        # TODO: check if has constraints
        if self._has_constr:
            self._raw_system = sp.sparse.vstack(
                [
                    sp.sparse.hstack([P + sp.sparse.diags(self._rho_vec), A.T]),
                    sp.sparse.hstack([A, reg * sp.sparse.eye(self._constr_dim)]),
                ]
            )
        else:
            self._raw_system = P + sp.sparse.diags(self._rho_vec)

        self._fac_system = qdldl.Solver(self._raw_system)

    def solve(self, rhs):
        return self._fac_system.solve(rhs)

    def update_rho(self, new_rho_vec):
        if self._has_constr:
            I0_matrix = sp.sparse.block_diag(
                [
                    sp.sparse.diags(new_rho_vec - self._rho_vec, format="csc"),
                    sp.sparse.csc_matrix((self._constr_dim, self._constr_dim)),
                ]
            )
        else:
            I0_matrix = sp.sparse.diags(new_rho_vec - self._rho_vec, format="csc")

        self._raw_system += I0_matrix
        self._fac_system.update(self._raw_system)

        self._rho_vec = np.copy(new_rho_vec)

    def reset(self):
        del self._raw_system
        del self._fac_system


class AbstractKKT:
    def __init__(self, P, A, rho_controller):
        self._dim = A.shape[1]
        self._constr_dim = A.shape[0]
        self.shape = (self._dim, self._constr_dim)
        self._rho_controller = rho_controller
        self._rho_vec = np.copy(rho_controller.get_rho_vec())

        self._P = P
        self._A = A
        self._reg = 0  # TODO: need this?
        self._splinop = sp.sparse.linalg.LinearOperator(
            (self._dim + self._constr_dim, self._dim + self._constr_dim),
            matvec=self.matvec,
            rmatvec=self.matvec,
        )

    def matvec(self, v):
        res = np.zeros(self._dim + self._constr_dim)
        res[: self._dim] += self._P @ v[: self._dim] + self._rho_vec * v[: self._dim]
        res[: self._dim] += self._A.rmatvec(v[self._dim :])
        res[self._dim :] += self._A.matvec(v[: self._dim])
        res[self._dim :] += self._reg * v[self._dim :]
        return res

    def solve(self, rhs):
        return sp.sparse.linalg.gmres(self._splinop, rhs, atol="legacy")[0]

    def update_rho(self, new_rho_vec):
        self._rho_vec = new_rho_vec
        # TODO: is below necessary or done already?
        # i.e. does rho update propagate?
        self._splinop = sp.sparse.linalg.LinearOperator(
            (self._dim + self._constr_dim, self._dim + self._constr_dim),
            matvec=self.matvec,
            rmatvec=self.matvec,
        )

    def reset(self):
        return


def build_kkt(reg, rho, P, A, dim, constr_dim, **kwargs):
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
