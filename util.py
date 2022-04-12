import numpy as np

# Separable functions // proximal operator:
# 0: f(x) = 0 // prox(v) = v
# 1: f(x) = |x| // prox(v) = prox(v) = S_{1/rho}(v)

# 0: f(x) = 0
def g_zero(v):
    return 0


def prox_zero(rho, v):
    return v


# 1: f(x) = |x|
def g_abs(v):
    return np.abs(v)


def prox_abs(rho, v):
    return np.maximum(v - 1 / rho, 0) - np.maximum(-v - 1 / rho, 0)


# 2: f(x) = I(x >= 0)
def g_pos(v):
    valid = np.all(v >= 0)
    if valid:
        return 0
    else:
        return np.inf


def prox_pos(rho, v):
    y = v
    y[np.where(v < 0)] = 0
    return y


g_funcs = [g_zero, g_abs, g_pos]
prox_ops = [prox_zero, prox_abs, prox_pos]


def apply_g_funcs(g, x):
    y = np.zeros(len(x))
    for i in range(len(g_funcs)):
        idx = np.where(g == i)
        y[idx] = g_funcs[i](x[idx])
    return np.sum(y)


def apply_prox_ops(rho, g, x):
    y = np.zeros(len(x))
    for i in range(len(prox_ops)):
        idx = np.where(g == i)
        y[idx] = prox_ops[i](rho[idx], x[idx])
    return y
