import numpy as np
import scipy as sp
import time

from zmq import has
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
    kkt_info,
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
        I0_matrix = sp.sparse.block_diag(
            [
                sp.sparse.identity(dim, format="csc"),
                sp.sparse.csc_matrix((constr_dim, constr_dim)),
            ]
        )

        if has_constr:
            kkt_info["quad_kkt_unreg"] += (new_rho_candidate - rho) * I0_matrix

        kkt_info["quad_kkt"] += (new_rho_candidate - rho) * I0_matrix
        kkt_info["F"].update(kkt_info["quad_kkt"])

        rho = new_rho_candidate

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


def admm(
    data,
    kkt_info,
    x,
    y,
    equil_scaling,
    obj_scale,
    rho,
    adaptive_rho,
    alpha,
    eps_abs,
    eps_rel,
    use_iter_refinement,
    verbose,
    max_iter,
    **kwargs
):
    if verbose:
        print(" #####     Beginning ADMM solve     #####")
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

    # ADMM iterates
    xk = x
    zk = x
    uk = y / rho
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
            if use_iter_refinement:
                kkt_solve = matrix.ir_solve(
                    kkt_info["quad_kkt_unreg"],
                    kkt_info["F"],
                    np.concatenate([-q + rho * (zk - uk), b]),
                )
                xk1 = kkt_solve[:dim]
                nuk1 = kkt_solve[dim:]
            else:
                kkt_solve = kkt_info["F"].solve(
                    np.concatenate([-q + rho * (zk - uk), b])
                )
                xk1 = kkt_solve[:dim]
                nuk1 = kkt_solve[dim:]
        else:
            xk1 = F.solve(-q + rho * (zk - uk))

        # Update z
        zk1 = proximal.apply_prox_ops(
            rho / obj_scale, equil_scaling, g, alpha * xk1 + (1 - alpha) * zk + uk
        )

        # Update u
        uk1 = uk + alpha * xk1 + (1 - alpha) * zk - zk1

        # Calculate residuals and objective
        r_prim = np.linalg.norm(xk1 - zk1, ord=np.inf)
        r_dual = np.linalg.norm(rho * (zk - zk1), ord=np.inf)
        obj_val = util.evaluate_objective(P, q, r, g, zk1, obj_scale, equil_scaling)

        # Check if we should print current status
        if verbose and (iter_num == 1 or iter_num == max_iter or iter_num % 25 == 0):
            util.print_status(
                iter_num,
                obj_val,
                r_prim,
                r_dual,
                rho,
                admm_start_time,
            )

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

            # Print status of this last iteration if we haven't already
            if verbose and (iter_num != max_iter and iter_num % 25 != 0):
                util.print_status(
                    iter_num, obj_val, r_prim, r_dual, rho, admm_start_time
                )

            if verbose:
                util.print_footer()
                print(
                    "Average",
                    (time.time() - admm_start_time - total_refactorization_time)
                    / iter_num,
                    "seconds per iteration",
                )
                print("Refactored {} times.".format(refactorization_count))
                print(
                    "Spent total {} seconds refactorizing.".format(
                        total_refactorization_time
                    )
                )

        # Update rho
        if adaptive_rho and (not finished) and (iter_num % 10 == 0):
            rho, refactorization_count, total_refactorization_time = update_rho(
                dim,
                has_constr,
                constr_dim,
                refactorization_count,
                total_refactorization_time,
                kkt_info,
                rho,
                r_prim,
                r_dual,
                xk1,
                zk1,
                uk1,
            )

        xk = xk1
        zk = zk1
        uk = uk1

    return {
        "x": equil_scaling * zk1,
        "y": rho * uk1,  # TODO: Does y need to be scaled by equil_scaling or obj_scale?
        "obj_val": util.evaluate_objective(P, q, r, g, zk1, obj_scale, equil_scaling),
    }
