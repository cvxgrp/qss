import numpy as np
import scipy as sp
import time
from qss import proximal

PRINT_WIDTH = 63
BULLET_WIDTH = 32


def evaluate_stop_crit(
    zk1,
    uk1,
    nuk1,
    dim,
    rho,
    eps_abs,
    eps_rel,
    P,
    q,
    A,
    b,
    equil_scaling,
    constr_scaling,
    obj_scale,
    ord=2,
):
    rprim = np.linalg.norm((A @ zk1 - b) / constr_scaling, ord=ord)
    rdual = np.linalg.norm(
        (P @ zk1 + q + A.T @ nuk1 + rho * uk1) / equil_scaling / obj_scale, ord=ord
    )
    epri = np.sqrt(dim) * eps_abs + eps_rel * max(
        np.linalg.norm(A @ zk1 / constr_scaling, ord=ord),
        np.linalg.norm(b / constr_scaling, ord=ord),
    )
    edual = np.sqrt(dim) * eps_abs + eps_rel / obj_scale * max(
        np.linalg.norm(P @ zk1 / equil_scaling, ord=ord),
        np.linalg.norm(q / equil_scaling, ord=ord),
        np.linalg.norm(A.T @ nuk1 / equil_scaling, ord=ord),
        np.linalg.norm(rho * uk1 / equil_scaling, ord=ord),
    )

    if rprim < epri and rdual < edual:
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


def print_status(iter_num, obj_val, r_prim, r_dual, rho, solve_start_time):
    print(
        "{} | {}  {}  {}  {}  {}".format(
            str(iter_num).rjust(5),
            format(obj_val, ".2e").ljust(10),
            format(np.linalg.norm(r_prim), ".2e").ljust(11),
            format(np.linalg.norm(r_dual), ".2e").ljust(9),
            format(rho, ".2e").ljust(6),
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
