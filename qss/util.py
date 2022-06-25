import numpy as np
import scipy as sp
import time
from qss import proximal


def evaluate_stop_crit(xk1, zk, zk1, uk1, dim, rho, eps_abs, eps_rel, P, q, ord=2):
    if ord == 2:
        epri = np.sqrt(dim) * eps_abs + eps_rel * max(
            np.linalg.norm(xk1, ord=2), np.linalg.norm(zk1, ord=2)
        )
        edual = np.sqrt(dim) * eps_abs + eps_rel * np.linalg.norm(rho * uk1, ord=2)
        if (
            np.linalg.norm(xk1 - zk1, ord=2) < epri
            and np.linalg.norm(rho * (zk - zk1), ord=2) < edual
        ):
            return True
        return False

    elif ord == np.inf:
        epri = eps_abs + eps_rel * max(
            np.linalg.norm(xk1, ord=np.inf), np.linalg.norm(zk1, ord=np.inf)
        )
        edual = eps_abs + eps_rel * max(
            np.linalg.norm(P @ xk1, ord=np.inf),
            np.linalg.norm(rho * uk1, ord=np.inf),
            np.linalg.norm(q, ord=np.inf),
        )
        if (
            np.linalg.norm(xk1 - zk1, ord=np.inf) < epri
            and np.linalg.norm(P @ xk1 + q + rho * uk1, ord=np.inf) < edual
        ):
            return True
        return False


def print_info():
    print("---------------------------------------------------------------")
    print("              QSS: the Quadratic-Separable Solver              ")
    print("                     author: Luke Volpatti                     ")
    print("---------------------------------------------------------------")
    print()


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
    return (0.5 * zk1 @ P @ zk1 + q @ zk1 + r) / obj_scale + proximal.apply_g_funcs(
        g, equil_scaling * zk1
    )
