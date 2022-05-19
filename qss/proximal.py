import numpy as np

# f(x) = 0
def g_zero(v, args):
    return np.zeros(len(v))


def prox_zero(rho, v, args):
    return v


def subdiff_zero(v, args):
    return np.zeros(len(v)), np.zeros(len(v))


# f(x) = |x|
def g_abs(v, args):
    return np.abs(v)


def prox_abs(rho, v, args):
    return np.maximum(v - 1 / rho, 0) - np.maximum(-v - 1 / rho, 0)


def subdiff_abs(v, args):
    ls = np.zeros(len(v))
    rs = np.zeros(len(v))

    ls[v < 0] = -1
    rs[v < 0] = -1

    ls[v > 0] = 1
    rs[v > 0] = 1

    ls[np.isclose(v, 0)] = -1
    rs[np.isclose(v, 0)] = 1

    return ls, rs


# f(x) = I(x >= 0)
def g_is_pos(v, args):
    return np.where(v >= 0, 0, np.inf)


def prox_is_pos(rho, v, args):
    return np.where(v < 0, 0, v)


def subdiff_is_pos(v, args):
    ls = np.zeros(len(v))
    rs = np.zeros(len(v))

    ls[v < 0] = np.nan
    rs[v < 0] = np.nan

    ls[v == 0] = -np.inf
    rs[v == 0] = 0

    return ls, rs


# f(x) = I(x <= 0)
def g_is_neg(v, args):
    return np.where(v <= 0, 0, np.inf)


def prox_is_neg(rho, v, args):
    return np.where(v > 0, 0, v)


def subdiff_is_neg(v, args):
    ls = np.zeros(len(v))
    rs = np.zeros(len(v))

    ls[v > 0] = np.nan
    rs[v > 0] = np.nan

    ls[v == 0] = 0
    rs[v == 0] = np.inf

    return ls, rs


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
    return np.where((v >= lb) & (v <= ub), 0, np.inf)


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
    output = np.where(v >= ub, ub, v)
    output = np.where(output <= lb, lb, output)
    return output


def subdiff_is_bound(v, args):
    # TODO: make sure lb < ub
    if "lb" in args:
        lb = args["lb"]
    else:
        lb = 0
    if "ub" in args:
        ub = args["ub"]
    else:
        ub = 1

    ls = np.zeros(len(v))
    rs = np.zeros(len(v))

    ls[v > ub] = np.nan
    rs[v > ub] = np.nan

    ls[v == ub] = 0
    rs[v == ub] = np.inf

    ls[v < lb] = np.nan
    rs[v < lb] = np.nan

    ls[v == lb] = -np.inf
    rs[v == lb] = 0

    return ls, rs


# f(x) = I(x == 0)
def g_is_zero(v, args):
    return np.where(v == 0, 0, np.inf)


def prox_is_zero(rho, v, args):
    return np.zeros(len(v))


def subdiff_is_zero(v, args):
    ls = np.nan * np.ones(len(v))
    rs = np.nan * np.ones(len(v))

    ls[v == 0] = -np.inf
    rs[v == 0] = np.inf

    return ls, rs


# f(x) = max{x, 0}
def g_pos(v, args):
    return np.maximum(v, 0)


def prox_pos(rho, v, args):
    output = np.where(v <= 0, v, 0)
    output = np.where(v > 1 / rho, v - 1 / rho, output)
    return output


def subdiff_pos(v, args):
    ls = np.zeros(len(v))
    rs = np.zeros(len(v))

    ls[v > 0] = 1
    rs[v > 0] = 1

    # TODO: change this to is_close?
    ls[v == 0] = 0
    rs[v == 0] = 1

    return ls, rs


# f(x) = max{-x, 0}
def g_neg(v, args):
    return np.maximum(-v, 0)


def prox_neg(rho, v, args):
    return np.where(v < -1 / rho, v + 1 / rho, v)


def subdiff_neg(v, args):
    ls = np.zeros(len(v))
    rs = np.zeros(len(v))

    ls[v < 0] = -1
    rs[v < 0] = -1

    # TODO: change this to is_close?
    ls[v == 0] = -1
    rs[v == 0] = 0

    return ls, rs


# f(x) = {0 if x == 0, 1 else}
def g_card(v, args):
    # TODO: change this to isclose?
    return np.where(v == 0, 0, 1)


def prox_card(rho, v, args):
    return np.where(np.abs(v) < np.sqrt(2 / rho), 0, v)


def subdiff_card(v, args):
    ls = np.nan * np.ones(len(v))
    rs = np.nan * np.ones(len(v))

    ls[v == 0] = 0
    rs[v == 0] = 0

    return ls, rs


# f(x) = 0.5 * |x| + (tau - 0.5) * x
def g_quantile(v, args):
    if "tau" in args:
        tau = args["tau"]
    else:
        tau = 0.5
    return 0.5 * np.abs(v) + (tau - 0.5) * v


def prox_quantile(rho, v, args):
    if "tau" in args:
        tau = args["tau"]
    else:
        tau = 0.5
    v_mod = v + 1 / rho * (0.5 - tau)
    return np.where(
        np.abs(v_mod) <= 1 / (2 * rho), 0, v_mod - np.sign(v_mod) * 1 / (2 * rho)
    )


def subdiff_quantile(v, args):
    if "tau" in args:
        tau = args["tau"]
    else:
        tau = 0.5

    ls = np.zeros(len(v))
    rs = np.zeros(len(v))

    ls[v > 0] = tau
    rs[v > 0] = tau

    ls[v < 0] = tau - 1
    rs[v < 0] = tau - 1

    # TODO: change to is_close?
    ls[v == 0] = tau - 1
    rs[v == 0] = tau

    return ls, rs


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


def subdiff_huber(v, args):
    if "M" in args:
        M = args["M"]
    else:
        M = 1

    ls = np.zeros(len(v))
    rs = np.zeros(len(v))

    abs_v = np.abs(v)

    ls[abs_v <= M] = 2 * v[abs_v <= M]
    rs[abs_v <= M] = 2 * v[abs_v <= M]

    ls[v > M] = 2 * M
    rs[v > M] = 2 * M

    ls[v < M] = -2 * M
    rs[v < M] = -2 * M

    return ls, rs


# f(x) = I(x is an integer)
def g_is_int(v, args):
    return np.where(
        np.isclose(np.mod(v, 1), 0) | np.isclose(np.mod(v, 1), 1), 0, np.inf
    )
    # TODO: change this to something like np.isclose(v, np.rint(v))


def prox_is_int(rho, v, args):
    return np.rint(v)


def subdiff_is_int(v, args):
    ls = np.nan * np.ones(len(v))
    rs = np.nan * np.ones(len(v))

    int_indices = g_is_int(v, args)

    ls[int_indices == 0] = 0
    rs[int_indices == 0] = 0

    return ls, rs


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


def subdiff_is_finite_set(v, args):
    ls = np.nan * np.ones(len(v))
    rs = np.nan * np.ones(len(v))

    in_set_indices = g_is_finite_set(v, args)

    ls[in_set_indices == 0] = 0
    rs[in_set_indices == 0] = 0

    return ls, rs


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


def subdiff_is_bool(v, args):
    ls = np.nan * np.ones(len(v))
    rs = np.nan * np.ones(len(v))

    is_bool_indices = g_is_bool(v, args)

    ls[is_bool_indices == 0] = 0
    rs[is_bool_indices == 0] = 0

    return ls, rs


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
    "is_pos": subdiff_is_pos,
    "is_neg": subdiff_is_neg,
    "is_bound": subdiff_is_bound,
    "is_zero": subdiff_is_zero,
    "pos": subdiff_pos,
    "neg": subdiff_neg,
    "card": subdiff_card,
    "quantile": subdiff_quantile,
    "huber": subdiff_huber,
    "is_int": subdiff_is_int,
    "is_finite_set": subdiff_is_finite_set,
    "is_bool": subdiff_is_bool,
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


def get_subdiff(g_list, x, equil_scaling, obj_scale):
    ls = np.zeros(len(x))
    rs = np.zeros(len(x))
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

        subdiff_func = subdiffs[func_name]
        g_ls, g_rs = subdiff_func(
            scale * equil_scaling[start_index:end_index] * x[start_index:end_index]
            - shift,
            g["args"],
        )
        g_ls = obj_scale * equil_scaling[start_index:end_index] * weight * scale * g_ls
        g_rs = obj_scale * equil_scaling[start_index:end_index] * weight * scale * g_rs
        ls[start_index:end_index] = g_ls
        rs[start_index:end_index] = g_rs

    return ls, rs
