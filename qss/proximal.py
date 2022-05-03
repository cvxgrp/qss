import numpy as np

# Separable functions // proximal operator:
# 0: f(x) = 0 // prox(v) = v
# 1: f(x) = |x| // prox(v) = prox(v) = S_{1/rho}(v)

# f(x) = 0
def g_zero(v, args):
    return 0


def prox_zero(rho, v, args):
    return v


def subdiff_zero(v):
    return np.zeros(len(v))


# f(x) = |x|
def g_abs(v, args):
    return np.abs(v)


def prox_abs(rho, v, args):
    return np.maximum(v - 1 / rho, 0) - np.maximum(-v - 1 / rho, 0)


def subdiff_abs(v):
    c = np.zeros(len(v))
    r = np.zeros(len(v))

    c[v < 0] = -1
    c[v > 0] = 1
    c[v == 0] = 0

    r[v < 0] = 0
    r[v > 0] = 0
    r[v == 0] = 1

    return c, r


# f(x) = I(x >= 0)
def g_indge0(v, args):
    valid = np.all(v >= 0)
    if valid:
        return 0
    else:
        return np.inf


def prox_indge0(rho, v, args):
    y = v
    y[np.where(v < 0)] = 0
    return y


# f(x) = I(0 <= x <= 1)
def g_indbox01(v, args):
    valid = np.all(np.logical_and(v >= 0, v <= 1))
    if valid:
        return 0
    else:
        return np.inf


def prox_indbox01(rho, v, args):
    v[v >= 1] = 1
    v[v <= 0] = 0
    return v


# f(x) = I(x == 0)
def g_is_zero(v, args):
    valid = np.all(v == 0)
    if valid:
        return 0
    else:
        return np.inf


def prox_is_zero(rho, v, args):
    return np.zeros(len(v))


# f(x) = {0 if x == 0, 1 else}
def g_card(v, args):
    return np.count_nonzero(v)


def prox_card(rho, v, args):
    v[v < np.sqrt(2 / rho)] = 0
    return v


# f(x) = 0.5 * |x| + (tau - 0.5) * x
def g_quantile(v, args):
    if "tau" in args:
        tau = args["tau"]
    else:
        tau = 1
    return 0.5 * np.abs(v) + (tau - 0.5) * v
    
def prox_quantile(rho, v, args):
    if "tau" in args:
        tau = args["tau"]
    else:
        tau = 1
    output = np.zeros(len(v))
    upper_thresh = v - tau/rho > 0
    lower_thresh = v + (1+tau)/rho < 0
    output[upper_thresh] = v[upper_thresh] - tau/rho[upper_thresh]
    output[lower_thresh] = v[lower_thresh] + (1+tau)/rho[lower_thresh]
    return output


g_funcs = {
    "zero": g_zero,
    "abs": g_abs,
    "indge0": g_indge0,
    "indbox01": g_indbox01,
    "is_zero": g_is_zero,
    "card": g_card,
    "quantile": g_quantile,
}

prox_ops = {
    "zero": prox_zero,
    "abs": prox_abs,
    "indge0": prox_indge0,
    "indbox01": prox_indbox01,
    "is_zero": prox_is_zero,
    "card": prox_card,
    "quantile": prox_quantile,
}

subdiffs = {
    "zero": subdiff_zero,
    "abs": subdiff_abs,
}


def apply_g_funcs(g_list, x):
    y = np.zeros(len(x))
    for g in g_list:
        func_name = g["g"]
        if ("args" not in g) or (g["args"] is None): # TODO: don't check this every iter
            g["args"] = {}
        if "args" in g and "weight" in g["args"]:
            weight = g["args"]["weight"]
        else:
            weight = 1
        if "args" in g and "scale" in g["args"]:
            scale = g["args"]["scale"]
        else:
            scale = 1
        if "args" in g and "shift" in g["args"]:
            shift = g["args"]["shift"]
        else:
            shift = 0
        start_index, end_index = g["range"]

        func = g_funcs[func_name]
        y[start_index:end_index] = weight * func(
            scale * x[start_index:end_index] - shift, g["args"]
        )
    return np.sum(y)


def apply_prox_ops(rho, equil_scaling, g_list, x):
    for g in g_list:
        prox_op_name = g["g"]
        if ("args" not in g) or (g["args"] is None): # TODO: don't check this every iter
            g["args"] = {}
        if "args" in g and "weight" in g["args"]:
            weight = g["args"]["weight"]
        else:
            weight = 1
        if "args" in g and "scale" in g["args"]:
            scale = g["args"]["scale"]
        else:
            scale = 1
        if "args" in g and "shift" in g["args"]:
            shift = g["args"]["shift"]
        else:
            shift = 0
        start_index, end_index = g["range"]

        new_scale = equil_scaling[start_index:end_index] * scale
        new_rho = rho / (weight * new_scale**2)

        prox = prox_ops[prox_op_name]
        x[start_index:end_index] = (
            prox(new_rho, new_scale * x[start_index:end_index] - shift, g["args"]) + shift
        ) / new_scale
    return x


def get_subdiff(g_list, x):
    c = np.zeros(len(x))
    r = np.zeros(len(x))
    for g in g_list:
        func_name = g["g"]
        # TODO: Shifting, scaling
        start_index, end_index = g["range"]

        subdiff_func = subdiffs[func_name]
        g_c, g_r = subdiff_func(x[start_index:end_index])
        c[start_index:end_index] = g_c
        r[start_index:end_index] = g_r

    return c, r
