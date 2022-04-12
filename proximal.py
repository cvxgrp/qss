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


g_funcs = {"zero": g_zero, "abs": g_abs, "indge0": g_indge0}
prox_ops = {"zero": prox_zero, "abs": prox_abs, "indge0": prox_indge0}

def apply_g_funcs(g_list, x):
    for g in g_list:
        func_name = g[0]
        shift_scale = g[1]
        start_index = g[2][0]
        end_index = g[2][1]

        func = g_funcs[func_name]
        x[start_index:end_index] = func(x[start_index:end_index])
    return np.sum(x)

def apply_prox_ops(rho, g_list, x):
    for g in g_list:
        prox_op_name = g[0]
        shift_scale = g[1]
        start_index = g[2][0]
        end_index = g[2][1]

        prox = prox_ops[prox_op_name]
        x[start_index:end_index] = prox(rho[start_index:end_index], x[start_index:end_index])
    return x
