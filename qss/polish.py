import numpy as np
import scipy as sp
import cvxpy as cp
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


def l1_descent_dir(g, x, P, q, r, equil_scaling, obj_scale):
    f_subdiff = P @ x + q
    ls, rs = proximal.get_subdiff(g, x, equil_scaling, obj_scale)

    ai = f_subdiff + ls
    bi = f_subdiff + rs

    # 1e-7 to be a bit more conservative when providing descent directions
    v_st = np.where(bi < -1e-7, 1, 0)
    v_st = np.where(ai > 1e-7, -1, v_st)

    # Calculate directional derivative
    dF_v = f_subdiff @ v_st + np.sum(np.maximum(v_st * ls, v_st * rs))

    return v_st, dF_v


def l2_descent_dir(g, x, P, q, r, equil_scaling, obj_scale):
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


def sd_eval_obj(x, v_st, a, b, c, t, g, equil_scaling, obj_scale):
    return (a + b * t + c * t**2) / obj_scale + proximal.apply_g_funcs(
        g, equil_scaling * (x + t * v_st)
    )


def steepest_descent(g, x, P, q, r, equil_scaling, obj_scale, x_obj, ord=2, max_iter=50):
    converged = False

    iter = 0
    prev_mid_t_obj = x_obj
    prev_t = 1

    while not converged and (iter < max_iter):
        iter += 1

        if ord == 1:
            v_st, dF_v = l1_descent_dir(g, x, P, q, r, equil_scaling, obj_scale)
        elif ord == 2: 
            v_st, dF_v = l2_descent_dir(g, x, P, q, r, equil_scaling, obj_scale)
        left_t = 0.5 * prev_t
        mid_t = prev_t
        right_t = 2 * prev_t

        a = 0.5 * x @ P @ x + q @ x + r
        b = x @ P @ v_st + q @ v_st
        c = 0.5 * v_st @ P @ v_st

        left_t_obj = sd_eval_obj(x, v_st, a, b, c, left_t, g, equil_scaling, obj_scale)
        mid_t_obj = sd_eval_obj(x, v_st, a, b, c, mid_t, g, equil_scaling, obj_scale)
        right_t_obj = sd_eval_obj(
            x, v_st, a, b, c, right_t, g, equil_scaling, obj_scale
        )

        found_t = False
        while not found_t:
            if mid_t_obj <= left_t_obj and mid_t_obj <= right_t_obj:
                found_t = True
            else:
                if mid_t_obj > left_t_obj:
                    right_t = mid_t
                    right_t_obj = mid_t_obj
                    mid_t = left_t
                    mid_t_obj = left_t_obj
                    left_t = 0.5 * left_t
                    left_t_obj = sd_eval_obj(
                        x, v_st, a, b, c, left_t, g, equil_scaling, obj_scale
                    )
                elif mid_t_obj > right_t_obj:
                    left_t = mid_t
                    left_t_obj = mid_t_obj
                    mid_t = right_t
                    mid_t_obj = right_t_obj
                    right_t *= 2
                    right_t_obj = sd_eval_obj(
                        x, v_st, a, b, c, right_t, g, equil_scaling, obj_scale
                    )

        x = x + mid_t * v_st
        if prev_mid_t_obj - mid_t_obj < 1e-5:
            converged = True
        else:
            prev_mid_t_obj = mid_t_obj

    return x, iter
