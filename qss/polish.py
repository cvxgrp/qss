import numpy as np
import scipy as sp
import cvxpy as cp
import time
from qss import proximal
from qss import util


def polish(g, zk1, P, q, r, A, b, equil_scaling, obj_scale, dim):
    subdiff_center, subdiff_radius = proximal.get_subdiff(g, zk1)
    fixed_x_indices = subdiff_radius != 0

    x = cp.Variable(dim)
    objective = cp.Minimize(
        (0.5 * cp.quad_form(x, P) + q @ x + r) / obj_scale
        + cp.multiply(x, equil_scaling) @ subdiff_center
    )
    constraints = [
        A @ x == b,
        x[fixed_x_indices] == zk1[fixed_x_indices],
    ]
    prob = cp.Problem(objective, constraints)
    prob.solve()

    zk1_polish = x.value
    return zk1_polish


def l2_descent_dir(g, x, P, q, r, equil_scaling, obj_scale):
    # TODO: do something with obj_scaling?
    # This will affect the scaling of v
    f_subdiff = P @ x + q
    ls, rs = proximal.get_subdiff(g, x, equil_scaling, obj_scale)

    ai = f_subdiff + ls
    bi = f_subdiff + rs

    # 1e-7 to be a bit more conservative when providing descent directions
    v_st = np.where(bi < -1e-7, -bi, 0)
    v_st = np.where(ai > 1e-7, -ai, v_st)

    # Calculate directional derivative
    dF_v = f_subdiff @ v_st + np.sum(np.maximum(v_st * ls, v_st * rs))

    return v_st, dF_v


def steepest_descent(g, x, P, q, r, equil_scaling, obj_scale):
    converged = False

    start_time = time.time()
    iter = 0
    while not converged:
        iter += 1

        v_st, dF_v = l2_descent_dir(g, x, P, q, r, equil_scaling, obj_scale)
        t = 1

        curr_obj_val = util.evaluate_objective(P, q, r, g, x, obj_scale, equil_scaling)
        desc_obj_val = util.evaluate_objective(
            P, q, r, g, x + t * v_st, obj_scale, equil_scaling
        )

        if desc_obj_val > curr_obj_val:
            # Need to keep halving until we find a step length that gets lower
            # objective.
            while desc_obj_val > curr_obj_val:
                t *= 0.5
                desc_obj_val = util.evaluate_objective(
                    P, q, r, g, x + t * v_st, obj_scale, equil_scaling
                )
        else:
            # We've found a descent step. Now see if increasing step size
            # results in even lower objective value.
            t_new = t
            new_desc_obj_val = desc_obj_val
            while new_desc_obj_val < desc_obj_val:
                t_new = 2 * t
                new_desc_obj_val = util.evaluate_objective(
                    P, q, r, g, x + t_new * v_st, obj_scale, equil_scaling
                )

        x = x + t * v_st
        if curr_obj_val - desc_obj_val < 1e-7:
            converged = True

    return x, time.time() - start_time, iter
