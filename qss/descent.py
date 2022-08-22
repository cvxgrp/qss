import numpy as np
import scipy as sp
import cvxpy as cp
import qdldl
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
    ls, rs = g.subdiff(equil_scaling, obj_scale, x)

    ai = f_subdiff + ls
    bi = f_subdiff + rs

    # 1e-7 to be a bit more conservative when providing descent directions
    v_st = np.where(bi < -1e-7, -bi, 0)
    v_st = np.where(ai > 1e-7, -ai, v_st)

    # Calculate directional derivative
    dF_v = f_subdiff @ v_st + np.sum(np.maximum(v_st * ls, v_st * rs))

    return v_st, dF_v


def sd_eval_obj(x, v_st, a, b, c, t, g, equil_scaling, obj_scale):
    return (a + b * t + c * t**2) / obj_scale + g.evaluate(
        equil_scaling * (x + t * v_st)
    )


def steepest_descent(g, x, P, q, r, equil_scaling, obj_scale, ord=2, max_iter=50):
    converged = False

    iter = 0
    prev_mid_t_obj = np.inf  # TODO: is it ok to start this with 0?
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
        prev_t = mid_t
        if prev_mid_t_obj - mid_t_obj < 1e-5:  # TODO: better stopping crit
            converged = True
        else:
            prev_mid_t_obj = mid_t_obj

    return x, iter


def do_line_search(x, v_st, prev_t, P, q, r, g, equil_scaling, obj_scale):
    left_t = 0.5 * prev_t
    mid_t = prev_t
    right_t = 2 * prev_t

    at = 0.5 * x @ P @ x + q @ x + r
    bt = x @ P @ v_st + q @ v_st
    ct = 0.5 * v_st @ P @ v_st

    left_t_obj = sd_eval_obj(x, v_st, at, bt, ct, left_t, g, equil_scaling, obj_scale)
    mid_t_obj = sd_eval_obj(x, v_st, at, bt, ct, mid_t, g, equil_scaling, obj_scale)
    right_t_obj = sd_eval_obj(x, v_st, at, bt, ct, right_t, g, equil_scaling, obj_scale)

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
                    x, v_st, at, bt, ct, left_t, g, equil_scaling, obj_scale
                )
            elif mid_t_obj > right_t_obj:
                left_t = mid_t
                left_t_obj = mid_t_obj
                mid_t = right_t
                mid_t_obj = right_t_obj
                right_t *= 2
                right_t_obj = sd_eval_obj(
                    x, v_st, at, bt, ct, right_t, g, equil_scaling, obj_scale
                )

    x = x + mid_t * v_st
    prev_t = mid_t

    return x, prev_t


def proj_sd(
    data,
    kkt_info,
    x,
    y,
    equil_scaling,
    obj_scale,
    descent_method,
    line_search,
    verbose,
    max_iter,
    ord=2,
    **kwargs
):
    if verbose:
        print(" ##### .    Beginning projected steepest descent solve     #####")
        util.print_header()
        proj_sd_start_time = time.time()

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

    iter_num = 0
    prev_mid_t_obj = np.inf
    prev_t = 1
    finished = False

    if descent_method == "momentum":
        momentum = 0.9
        prev_step = np.zeros(dim)
    elif descent_method == "adam":
        beta1 = 0.9
        beta2 = 0.999
        beta1t = beta1
        beta2t = beta2
        m_adam = 0
        v_adam = 0

    new_kkt = sp.sparse.vstack(
        [
            sp.sparse.hstack([sp.sparse.identity(dim), A.T]),
            sp.sparse.hstack([A, 1e-7 * sp.sparse.eye(len(b))]),
        ]
    )
    F = qdldl.Solver(new_kkt)

    while (not finished) and (iter_num < max_iter):
        iter_num += 1

        if ord == 1:
            v_st, dF_v = l1_descent_dir(g, x, P, q, r, equil_scaling, obj_scale)
        elif ord == 2:
            v_st_np, dF_v = l2_descent_dir(g, x, P, q, r, equil_scaling, obj_scale)

        # v_st = F.solve(np.concatenate([P @ v_st_np - q, np.zeros_like(b_constr)]))[:dim]
        v_st = F.solve(np.concatenate([v_st_np, np.zeros_like(b)]))[:dim]

        if descent_method == "momentum":
            v_st = momentum * prev_step + v_st
            prev_step = v_st
            x, prev_t = do_line_search(
                x, v_st, prev_t, P, q, r, g, equil_scaling, obj_scale
            )

        elif descent_method == "adam":
            v_st = -v_st
            m_adam = beta1 * m_adam + (1 - beta1) * v_st
            v_adam = beta2 * v_adam + (1 - beta2) * v_st * v_st
            mhat = m_adam / (1 - beta1t)
            vhat = v_adam / (1 - beta2t)
            beta1t *= beta1
            beta2t *= beta2
            v_st = -mhat / (np.sqrt(vhat) + 1e-8)
            if line_search:
                x, prev_t = do_line_search(
                    x, v_st, prev_t, P, q, r, g, equil_scaling, obj_scale
                )
            else:
                x = x + 0.001 * v_st

        # if prev_mid_t_obj - mid_t_obj < 1e-7:  # TODO: better stopping crit
        #     finished = True
        # else:
        #     prev_mid_t_obj = mid_t_obj

        if verbose and (
            finished or iter_num == 1 or iter_num == max_iter or iter_num % 25 == 0
        ):
            obj_val = util.evaluate_objective(P, q, r, g, x, obj_scale, equil_scaling)
            util.print_status(
                iter_num,
                obj_val,
                np.linalg.norm(A @ x - b),
                np.array([-1]),
                -1,
                proj_sd_start_time,
            )

    return {
        "x": x,
        "y": -(
            P @ x + q
        ),  # TODO: Does y need to be scaled by equil_scaling or obj_scale?
        "obj_val": util.evaluate_objective(P, q, r, g, x, obj_scale, equil_scaling),
    }
