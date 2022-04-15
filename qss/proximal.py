import numpy as np

# Separable functions // proximal operator:
# 0: f(x) = 0 // prox(v) = v
# 1: f(x) = |x| // prox(v) = prox(v) = S_{1/rho}(v)

# f(x) = 0
def g_zero(v):
    return 0


def prox_zero(rho, v):
    return v


# f(x) = |x|
def g_abs(v):
    return np.abs(v)


def prox_abs(rho, v):
    return np.maximum(v - 1 / rho, 0) - np.maximum(-v - 1 / rho, 0)


# f(x) = I(x >= 0)
def g_indge0(v):
    valid = np.all(v >= 0)
    if valid:
        return 0
    else:
        return np.inf


def prox_indge0(rho, v):
    y = v
    y[np.where(v < 0)] = 0
    return y


# f(x) = I(0 <= x <= 1)
def g_indbox01(v):
    valid = np.all(np.logical_and(v >= 0, v <= 1))
    if valid:
        return 0
    else:
        return np.inf


def prox_indbox01(rho, v):
    v[v >= 1] = 1
    v[v <= 0] = 0
    return v


g_funcs = {"zero": g_zero, "abs": g_abs, "indge0": g_indge0, "indbox01": g_indbox01}

prox_ops = {
    "zero": prox_zero,
    "abs": prox_abs,
    "indge0": prox_indge0,
    "indbox01": prox_indbox01,
}


def apply_g_funcs(g_list, x):
    y = np.zeros(len(x))
    for g in g_list:
        func_name = g[0]
        if g[1] == []:
            t = 1
            a = 1
            b = 0
        else:
            t, a, b = g[1]
        start_index = g[2][0]
        end_index = g[2][1]

        func = g_funcs[func_name]
        y[start_index:end_index] = t * func(a * x[start_index:end_index] - b)
    return np.sum(y)


def apply_prox_ops(rho, equil_scaling, g_list, x):
    for g in g_list:
        prox_op_name = g[0]
        if g[1] == []:
            t = 1
            a = 1
            b = 0
        else:
            t, a, b = g[1]
        start_index = g[2][0]
        end_index = g[2][1]

        new_a = equil_scaling[start_index:end_index] * a
        new_rho = rho / (t * new_a**2)

        prox = prox_ops[prox_op_name]
        x[start_index:end_index] = (
            prox(new_rho, new_a * x[start_index:end_index] - b) + b
        ) / new_a
    return x
