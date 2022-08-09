from functools import partial
import numpy as np
import scipy as sp
import time
from qss import proximal

PRINT_WIDTH = 63
BULLET_WIDTH = 32


def evaluate_stop_crit(xk1, zk, zk1, uk1, dim, rho, eps_abs, eps_rel, ord=2):
    if ord == 2:
        func = partial(np.linalg.norm, ord=2)
    elif ord == np.inf:
        func = partial(np.linalg.norm, ord=np.inf)
    epri = np.sqrt(dim) * eps_abs + eps_rel * max(func(xk1), func(zk1))
    edual = np.sqrt(dim) * eps_abs + eps_rel * func(rho * uk1)
    if (func(xk1 - zk1) < epri and func(rho * (zk - zk1)) < edual):
        return True
    else:
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
