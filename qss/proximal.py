import numpy as np

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
    c[np.isclose(v, 0)] = 0

    r[v < 0] = 0
    r[v > 0] = 0
    r[np.isclose(v, 0)] = 1

    return c, r


# f(x) = I(x >= 0)
def g_is_pos(v, args):
    valid = np.all(v >= 0)  # TODO: don't do np.all. Separable
    if valid:
        return 0
    else:
        return np.inf


def prox_is_pos(rho, v, args):
    y = v
    y[np.where(v < 0)] = 0
    return y


# f(x) = I(x <= 0)
def g_is_neg(v, args):
    valid = np.all(v <= 0)  # TODO: should be doing this separable??
    if valid:
        return 0
    else:
        return np.inf


def prox_is_neg(v, args):
    y = v
    y[np.where(v > 0)] = 0
    return y


# f(x) = I(0 <= x <= 1)
def g_is_bound(v, args):
    # TODO: make sure lb < ub
    if "lb" in args:
        lb = args["lb"]
    else:
        lb = 0
    if "ub" in args:
        ub = args["ub"]
    else:
        ub = 1
    valid = np.all(np.logical_and(v >= lb, v <= ub))
    if valid:
        return 0
    else:
        return np.inf
    # TODO: add upper and lower bound parameters


def prox_is_bound(rho, v, args):
    # TODO: make sure lb < ub
    if "lb" in args:
        lb = args["lb"]
    else:
        lb = 0
    if "ub" in args:
        ub = args["ub"]
    else:
        ub = 1
    v[v >= ub] = ub
    v[v <= lb] = lb
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


# f(x) = max{x, 0}
def g_pos(v, args):
    return np.maximum(v, 0)


def prox_pos(rho, v, args):
    return np.where(v <= 1 / rho, 0, v - 1 / rho)


# f(x) = max{-x, 0}
def g_neg(v, args):
    return np.maximum(-v, 0)


def prox_neg(rho, v, args):
    return np.where(v < -1 / rho, v + 1 / rho, 0)


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
    v_mod = v + 1 / rho * (0.5 - tau)
    return np.where(
        np.abs(v_mod) <= 1 / (2 * rho), 0, v_mod - np.sign(v_mod) * 1 / (2 * rho)
    )


# f(x) = huber(x)
def g_huber(v, args):
    if "M" in args:
        M = args["M"]
    else:
        M = 1
    abs_v = np.abs(v)
    return np.where(abs_v <= M, abs_v**2, 2 * M * abs_v - M * M)


def prox_huber(rho, v, args):
    if "M" in args:
        M = args["M"]
    else:
        M = 1
    return np.where(
        np.abs(v) <= M * (rho + 2) / rho,
        rho / (2 + rho) * v,
        v - np.sign(v) * 2 * M / rho,
    )


# f(x) = I(x is an integer)
def g_is_int(v, args):
    return np.where(
        np.isclose(np.mod(v, 1), 0) | np.isclose(np.mod(v, 1), 1), 0, np.inf
    )
    # TODO: change this to something like np.isclose(v, np.rint(v))


def prox_is_int(rho, v, args):
    return np.rint(v)


# f(x; S) = I(x is in S)
def g_is_finite_set(v, args):
    S = np.array(list(args["S"]))
    is_almost_in_S = np.isclose(v.reshape((-1, 1)), S.reshape((1, -1))).any(axis=1)
    return np.where(is_almost_in_S, 0, np.inf)


def prox_is_finite_set(rho, v, args):
    S = np.array(list(args["S"]))
    v = np.asarray(v)
    diffs = np.subtract(v.reshape((-1, 1)), S.reshape((1, -1)))
    idx = np.argmin(np.abs(diffs), axis=1)
    return S[idx]


# f(x) = I(x in {0,1})
def g_is_bool(v, args):
    S = np.array([0, 1])
    is_almost_in_S = np.isclose(v.reshape((-1, 1)), S.reshape((1, -1))).any(axis=1)
    return np.where(is_almost_in_S, 0, np.inf)


def prox_is_bool(rho, v, args):
    S = np.array([0, 1])
    v = np.asarray(v)
    diffs = np.subtract(v.reshape((-1, 1)), S.reshape((1, -1)))
    idx = np.argmin(np.abs(diffs), axis=1)
    return S[idx]


g_funcs = {
    "zero": g_zero,
    "abs": g_abs,
    "is_pos": g_is_pos,
    "is_neg": g_is_neg,
    "is_bound": g_is_bound,
    "is_zero": g_is_zero,
    "pos": g_pos,
    "neg": g_neg,
    "card": g_card,
    "quantile": g_quantile,
    "huber": g_huber,
    "is_int": g_is_int,
    "is_finite_set": g_is_finite_set,
    "is_bool": g_is_bool,
}

prox_ops = {
    "zero": prox_zero,
    "abs": prox_abs,
    "is_pos": prox_is_pos,
    "is_neg": prox_is_neg,
    "is_bound": prox_is_bound,
    "is_zero": prox_is_zero,
    "pos": prox_pos,
    "neg": prox_neg,
    "card": prox_card,
    "quantile": prox_quantile,
    "huber": prox_huber,
    "is_int": prox_is_int,
    "is_finite_set": prox_is_finite_set,
    "is_bool": prox_is_bool,
}

subdiffs = {
    "zero": subdiff_zero,
    "abs": subdiff_abs,
}


def apply_g_funcs(g_list, x):
    y = np.zeros(len(x))
    for g in g_list:
        func_name = g["g"]
        if ("args" not in g) or (
            g["args"] is None
        ):  # TODO: don't check this every iter
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
        if ("args" not in g) or (
            g["args"] is None
        ):  # TODO: don't check this every iter
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
            prox(new_rho, new_scale * x[start_index:end_index] - shift, g["args"])
            + shift
        ) / new_scale
    return x


def get_subdiff(g_list, x):
    c = np.zeros(len(x))
    r = np.zeros(len(x))
    for g in g_list:
        func_name = g["g"]
        # TODO: Shifting, scaling
        if "args" in g and "weight" in g["args"]:
            weight = g["args"]["weight"]
        else:
            weight = 1
        start_index, end_index = g["range"]

        subdiff_func = subdiffs[func_name]
        g_c, g_r = subdiff_func(x[start_index:end_index])
        c[start_index:end_index] = weight * g_c
        r[start_index:end_index] = weight * g_r

    return c, r
