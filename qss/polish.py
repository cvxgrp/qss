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
