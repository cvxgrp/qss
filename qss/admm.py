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
    rho,
    r_prim,
    r_dual,
    xk1,
    zk1,
    uk1,
):
    # Add 1e-30 to denominators to avoid divide by zero
    new_rho_candidate = rho * np.sqrt(
        r_prim
        / (r_dual + 1e-30)
        * np.linalg.norm(rho * uk1)
        / (
            max(
                np.linalg.norm(xk1, ord=np.inf),
                np.linalg.norm(zk1, ord=np.inf),
            )
            + 1e-30
        )
    )

    # This is for the first iteration
    if new_rho_candidate == 0:
        new_rho_candidate = rho

    # Check if new rho is different enough from old to warrant update
    if new_rho_candidate / rho > 5 or rho / new_rho_candidate > 5:
        refactorization_start_time = time.time()
        uk1 *= rho  # take back to yk1
        new_rho_candidate = min(max(new_rho_candidate, RHO_MIN), RHO_MAX)
        uk1 /= new_rho_candidate

        # Update KKT matrix
        kkt_system.update_rho(new_rho_candidate)

        return (
            rho,
            refactorization_count + 1,
            total_refactorization_time + time.time() - refactorization_start_time,
        )
    return (
        rho,
        refactorization_count,
        total_refactorization_time,
    )


def admm(data, kkt_system, options, x, y, equil_scaling, obj_scale, **kwargs):
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

    rho = options["rho"]
    adaptive_rho = options["adaptive_rho"]
    alpha = options["alpha"]
    eps_abs = options["eps_abs"]
    eps_rel = options["eps_rel"]
    use_iter_refinement = options["use_iter_refinement"]
    verbose = options["verbose"]
    max_iter = options["max_iter"]

    # ADMM iterates
    zk = x
    uk = y / rho  # TODO: do smth with equil_scaling/obj_scale here?
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

        # Update x
        if has_constr:
            kkt_solve = kkt_system.solve(np.concatenate([-q + rho * (zk - uk), b]))
            xk1 = kkt_solve[:dim]
            nuk1 = kkt_solve[dim:]
        else:
            xk1 = kkt_system.solve(-q + rho * (zk - uk))

        # Update z
        zk1 = g.prox(
            rho / obj_scale, equil_scaling, alpha * xk1 + (1 - alpha) * zk + uk
        )

        # Update u
        uk1 = uk + alpha * xk1 + (1 - alpha) * zk - zk1

        # Calculate residuals and objective
        r_prim = np.linalg.norm(xk1 - zk1, ord=np.inf)
        r_dual = np.linalg.norm(rho * (zk - zk1), ord=np.inf)
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
                rho,
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
                rho,
                admm_start_time,
            )

        # Update rho
        if adaptive_rho and (not finished) and (iter_num % 10 == 0):
            rho, refactorization_count, total_refactorization_time = update_rho(
                dim,
                has_constr,
                constr_dim,
                refactorization_count,
                total_refactorization_time,
                kkt_system,
                rho,
                r_prim,
                r_dual,
                xk1,
                zk1,
                uk1,
            )
            options["rho"] = rho  # Update globally

        xk = xk1
        zk = zk1
        uk = uk1

    iterates = {}
    iterates["x"] = zk1
    iterates["y"] = rho * uk1
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
