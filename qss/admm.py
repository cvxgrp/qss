import numpy as np
import scipy as sp
import time
from qss import proximal
from qss import matrix
from qss import util

# Constants
RHO_MIN = 1e-6
RHO_MAX = 1e6


def update_rho(
    dim,
    has_constr,
    constr_dim,
    refactorization_count,
    total_refactorization_time,
    kkt_system,
    rho_controller,
    xk1,
    zk,
    zk1,
    uk1,
):
    rho_vec = rho_controller.get_rho_vec()

    r_prim = xk1 - zk1
    r_dual = rho_vec * (zk - zk1)

    refactor = False
    for i, bool_range in enumerate(rho_controller._g.bool_ranges):
        local_new_rho_cand = np.sqrt(
            np.linalg.norm(r_prim[bool_range], ord=2)
            / (np.linalg.norm(r_dual[bool_range], ord=2) + 1e-30)
            * np.linalg.norm(rho_controller.rho_by_block[i] * uk1[bool_range])
            / (
                max(
                    np.linalg.norm(xk1[bool_range], ord=2),
                    np.linalg.norm(zk1[bool_range], ord=2),
                )
                + 1e-30
            )
        )

        local_new_rho_cand = min(max(local_new_rho_cand, RHO_MIN), RHO_MAX)

        if local_new_rho_cand / rho_controller.rho_by_block[i] > 5:
            # local_new_rho_cand = rho_controller.rho_by_block[i] * 5
            refactor = True
        elif rho_controller.rho_by_block[i] / local_new_rho_cand > 5:
            # local_new_rho_cand = rho_controller.rho_by_block[i] / 5
            refactor = True
        else:
            local_new_rho_cand = rho_controller.rho_by_block[i]

        uk1[bool_range] *= rho_controller.rho_by_block[i]
        rho_controller.rho_by_block[i] = local_new_rho_cand
        uk1[bool_range] /= rho_controller.rho_by_block[i]

    if refactor:
        refactorization_start_time = time.time()
        kkt_system.update_rho(rho_controller.get_rho_vec())
        return (
            refactorization_count + 1,
            total_refactorization_time + time.time() - refactorization_start_time,
        )
    return (
        refactorization_count,
        total_refactorization_time,
    )


def admm(
    data, kkt_system, options, rho_controller, x, y, equil_scaling, obj_scale, **kwargs
):
    if options["verbose"]:
        print("")
        print("ADMM solve".center(util.PRINT_WIDTH))
        util.print_header()
        admm_start_time = time.time()

    # Unpacking data
    P = data["P"]
    q = data["q"]
    r = data["r"]
    g = data["g"]
    A = data["A"]
    b = data["b"]
    dim = data["dim"]
    has_constr = data["has_constr"]
    constr_dim = data["constr_dim"]

    adaptive_rho = options["adaptive_rho"]
    alpha = options["alpha"]
    eps_abs = options["eps_abs"]
    eps_rel = options["eps_rel"]
    verbose = options["verbose"]
    max_iter = options["max_iter"]

    # ADMM iterates
    zk = x
    uk = (
        y / rho_controller.get_rho_vec()
    )  # TODO: do smth with equil_scaling/obj_scale here?
    # TODO: initialize uk = -q / rho?
    xk1 = np.zeros(dim)
    zk1 = np.zeros(dim)
    uk1 = np.zeros(dim)
    nuk1 = np.zeros(dim)

    iter_num = 0
    refactorization_count = 0
    total_refactorization_time = 0
    finished = False

    while not finished:
        iter_num += 1
        rho_vec = rho_controller.get_rho_vec()

        # Update x
        if has_constr:
            kkt_solve = kkt_system.solve(np.concatenate([-q + rho_vec * (zk - uk), b]))
            xk1 = kkt_solve[:dim]
            nuk1 = kkt_solve[dim:]
        else:
            xk1 = kkt_system.solve(-q + rho_vec * (zk - uk))

        # Update z
        zk1 = g.prox(
            rho_vec / obj_scale,
            equil_scaling,
            alpha * xk1 + (1 - alpha) * zk + uk,
        )

        # Update u
        uk1 = uk + alpha * xk1 + (1 - alpha) * zk - zk1

        # Calculate residuals and objective
        r_prim = np.linalg.norm(xk1 - zk1, ord=2)
        r_dual = np.linalg.norm(rho_vec * (zk - zk1), ord=2)
        obj_val = util.evaluate_objective(P, q, r, g, zk1, obj_scale, equil_scaling)

        # Check if we should stop
        if iter_num == max_iter or (
            iter_num % 10 == 0
            and util.evaluate_stop_crit(
                xk1,
                zk,
                zk1,
                uk1,
                dim,
                rho_vec,
                eps_abs,
                eps_rel,
                P,
                q,
                ord=2,
            )
        ):
            finished = True

        # Check if we should print current status
        if verbose and (
            finished or iter_num == 1 or iter_num == max_iter or iter_num % 25 == 0
        ):
            util.print_status(
                iter_num,
                obj_val,
                r_prim,
                r_dual,
                rho_controller.rho_by_block,
                admm_start_time,
            )

        # Update rho
        if adaptive_rho and (not finished) and ((iter_num + 1) % 50 == 0):
            refactorization_count, total_refactorization_time = update_rho(
                dim,
                has_constr,
                constr_dim,
                refactorization_count,
                total_refactorization_time,
                kkt_system,
                rho_controller,
                xk1,
                zk,
                zk1,
                uk1,
            )

        xk = xk1
        zk = zk1
        uk = uk1

    iterates = {}
    iterates["x"] = zk1
    iterates["y"] = rho_controller.get_rho_vec() * uk1
    iterates["obj_val"] = util.evaluate_objective(
        P, q, r, g, zk1, obj_scale, equil_scaling
    )

    if verbose:
        util.print_footer()
        print(
            "{} {}{}".format(
                "avg time per iter:".ljust(util.BULLET_WIDTH),
                format(
                    (time.time() - admm_start_time - total_refactorization_time)
                    / iter_num,
                    ".2e",
                ).ljust(5),
                "s",
            )
        )
        print(
            "{} {}".format(
                "refactorizations:".ljust(util.BULLET_WIDTH), str(refactorization_count)
            )
        )
        print(
            "{} {}{}".format(
                "total time spent refactorizing:".ljust(util.BULLET_WIDTH),
                format(total_refactorization_time, ".2e").ljust(5),
                "s",
            )
        )

    return iterates
