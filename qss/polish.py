import numpy as np
import scipy as sp
import cvxpy as cp
from qss import proximal


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


def l2_descent_dir(g, x, P, q, r, equil_scaling, obj_scale, dim):
    # TODO: do something with obj_scaling?
    # This will affect the scaling of v
    f_subdiff = (P @ x + q)
    ls, rs = proximal.get_subdiff(g, equil_scaling * x)
    ls *= obj_scale
    rs *= obj_scale

    ai = f_subdiff + ls
    bi = f_subdiff + rs
    
    # TODO: check bi < 1e-10 instead of 0?
    v_st = np.where(bi < -1e-7, -bi, 0)
    v_st = np.where(ai > 1e-7, -ai, v_st)

    # Calculate directional derivative
    dF_v = f_subdiff @ v_st + np.sum(np.maximum(v_st * ls, v_st * rs))

    return v_st, dF_v