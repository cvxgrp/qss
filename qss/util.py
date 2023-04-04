import numpy as np
import scipy as sp
import time
from qss import proximal
from qss import linearoperator

PRINT_WIDTH = 63
BULLET_WIDTH = 32


def evaluate_stop_crit(
        xk1,
        zk,
        zk1,
        uk1,
        nuk1,
        dim,
        rho_vec,
        eps_abs,
        eps_rel,
        P,
        q,
        A,
        b,
        crit="orig",
        ord=np.inf,
):
    if crit == "admm":
        r_prim = np.linalg.norm(xk1 - zk1, ord=ord)
        r_dual = np.linalg.norm(rho_vec * (zk - zk1), ord=ord)
        epri = eps_rel * max(np.linalg.norm(xk1, ord=ord), np.linalg.norm(zk1, ord=ord))
        edual = eps_rel * np.linalg.norm(rho_vec * uk1, ord=ord)

    elif crit == "orig":
        Azk1 = A @ zk1
        Pzk1 = P @ zk1
        ATnuk1 = A.T @ nuk1
        rhouk1 = rho_vec * uk1
        r_prim = np.linalg.norm(Azk1 - b, ord=ord)
        r_dual = np.linalg.norm(Pzk1 + q + ATnuk1 + rhouk1, ord=ord)
        epri = eps_rel * max(np.linalg.norm(Azk1, ord=ord), np.linalg.norm(b, ord=ord))
        edual = eps_rel * max(
            np.linalg.norm(Pzk1, ord=ord),
            np.linalg.norm(q, ord=ord),
            np.linalg.norm(ATnuk1, ord=ord),
            np.linalg.norm(rhouk1, ord=ord),
        )

    if ord == 2:
        epri += np.sqrt(dim) * eps_abs
        edual += np.sqrt(dim) * eps_abs
    else:
        epri += eps_abs
        edual += eps_abs

    if r_prim < epri and r_dual < edual:
        return True
    return False


def print_info():
    print("---------------------------------------------------------------")
    print("              QSS: the Quadratic-Separable Solver              ")
    print("                     author: Luke Volpatti                     ")
    print("---------------------------------------------------------------")


def print_header():
    print("---------------------------------------------------------------")
    print(" iter | objective | primal res | dual res |   rho   | time (s) ")
    print("---------------------------------------------------------------")


def print_footer():
    print("---------------------------------------------------------------")


def print_status(iter_num, obj_val, r_prim, r_dual, rho_vec, solve_start_time):
    print(
        "{} | {}  {}  {}  {}  {}".format(
            str(iter_num).rjust(5),
            format(obj_val, ".2e").ljust(10),
            format(np.linalg.norm(r_prim), ".2e").ljust(11),
            format(np.linalg.norm(r_dual), ".2e").ljust(9),
            # format(rho, ".2e").ljust(6),
            rho_vec,
            format(time.time() - solve_start_time, ".2e").ljust(8),
        )
    )


def evaluate_objective(P, q, r, g, zk1, obj_scale, equil_scaling):
    return (0.5 * zk1 @ P @ zk1 + q @ zk1 + r) / obj_scale + g.evaluate(
        equil_scaling * zk1
    )


def print_summary(obj_val, total_solve_time):
    print()
    print("{} {}".format("objective value:".ljust(32), obj_val))
    print("{} {:.4}s".format("total solve time:".ljust(32), total_solve_time, ".2e"))


class RhoController:
    def __init__(self, g, rho_init):
        # The last element of rho_by_block corresponds to zeros entries.
        # TODO: the below assumes that there are no "zero" g's.
        self.rho_by_block = rho_init * np.ones(len(g._g_list) + 1)
        # self.rho_by_block[0] = 1e10
        # self.rho_by_block[1] = 1e-10
        # self.rho_by_block = np.ones(len(g._g_list) + 1)
        # for i in range(len(self.rho_by_block)):
        #     if i < len(self.rho_by_block) - 1 and type(g._g_list[i]) is not qss.proximal.Zero:
        #         self.rho_by_block[i] = 1e6
        #     else:
        #         self.rho_by_block[i] = 1e-6
        # self.rho_by_block = 1e-6 * np.ones(len(g._g_list) + 1)
        # self.rho_by_block[-1] = 1e-6
        self._g = g

    def get_rho_vec(self):
        rho_vec = self.rho_by_block[-1] * np.ones(self._g.dim)

        for index, g in enumerate(self._g._g_list):
            start_index, end_index = g["range"]
            rho_vec[start_index:end_index] = self.rho_by_block[index]

        return rho_vec


def copy_problem_data(data, relax=False):
    data_copy = {}
    # Making copies of the input data and storing
    if type(data["P"]) is not linearoperator.LinearOperator:
        data_copy["P"] = data["P"].copy()
    else:
        data_copy["P"] = data["P"]
    data_copy["dim"] = data["P"].shape[0]
    data_copy["q"] = np.copy(data["q"])
    data_copy["r"] = data["r"]
    data_copy["g"] = proximal.GCollection(data["g"], data_copy["dim"], relax=relax)

    data_copy["abstract_constr"] = False
    if ("A" in data) and (type(data["A"]) is linearoperator.LinearOperator):
        data_copy["A"] = data["A"]  # TODO: in future, copy w/ linop .copy()
        data_copy["b"] = np.copy(data["b"])
        data_copy["abstract_constr"] = True
        data_copy["has_constr"] = True
        data_copy["constr_dim"] = data["A"].shape[0]
    elif ("A" not in data) or (data["A"] is None) or (data["A"].nnz == 0):
        # TODO: get rid of this placeholder when QSS is more object-oriented, i.e.,
        # when all problem data is passed around together.
        # I'm using the placeholder for now to avoid littering precondition.py with
        # 'if' statements.
        data_copy["A"] = sp.sparse.csc_matrix((1, data_copy["dim"]))
        data_copy["b"] = np.zeros(1)
        data_copy["has_constr"] = False
        data_copy["constr_dim"] = 1
    else:
        data_copy["A"] = data["A"].copy()
        data_copy["b"] = np.copy(data["b"])
        data_copy["has_constr"] = True
        data_copy["constr_dim"] = data["A"].shape[0]
    return data_copy
