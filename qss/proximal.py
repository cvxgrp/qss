import numpy as np
from abc import ABC, abstractmethod

G_FUNC_NAMES = {
    "zero",
    "abs",
    "is_pos",
    "is_neg",
    "is_bound",
    "is_zero",
    "pos",
    "neg",
    "card",
    "card_constr",
    "quantile",
    "huber",
    "is_int",
    "is_finite_set",
    "is_bool",
}


class G(ABC):
    def __init__(self, weight, scale, shift):
        self._weight = weight
        self._scale = scale
        self._shift = shift
        self._is_convex = None  # TODO: Instantiate this here?

    @abstractmethod
    def evaluate_raw(self, v):
        pass

    @abstractmethod
    def prox_raw(self, rho, v):
        pass

    @abstractmethod
    def subdiff_raw(self, v):
        pass

    def evaluate(self, v):
        return self._weight * self.evaluate_raw(self._scale * v - self._shift)

    def prox(self, rho, equil_scaling, v):
        new_scale = equil_scaling * self._scale
        new_rho = rho / (self._weight * new_scale**2)

        return (
            self.prox_raw(new_rho, new_scale * v - self._shift) + self._shift
        ) / new_scale

    def subdiff(self, equil_scaling, obj_scale, v):
        g_ls, g_rs = self.subdiff_raw(self._scale * equil_scaling * v - self._shift)
        g_ls *= obj_scale * equil_scaling * self._weight * self._scale
        g_rs *= obj_scale * equil_scaling * self._weight * self._scale

        return g_ls, g_rs


class Zero(G):
    def __init__(self, weight, scale, shift):
        super().__init__(weight, scale, shift)
        self._is_convex = True

    def evaluate_raw(self, v):
        return np.zeros(np.asarray(v).shape)

    def prox_raw(self, rho, v):
        return np.asarray(v)

    def subdiff_raw(self, v):
        v = np.asarray(v)
        return np.zeros(v.shape), np.zeros(v.shape)


class Abs(G):
    def __init__(self, weight, scale, shift):
        super().__init__(weight, scale, shift)
        self._is_convex = True

    def evaluate_raw(self, v):
        return np.abs(v)

    def prox_raw(self, rho, v):
        return np.maximum(v - 1 / rho, 0) - np.maximum(-v - 1 / rho, 0)

    def subdiff_raw(self, v):
        v = np.asarray(v)
        ls = np.zeros(v.shape)
        rs = np.zeros(v.shape)

        ls[v < 0] = -1
        rs[v < 0] = -1

        ls[v > 0] = 1
        rs[v > 0] = 1

        ls[np.isclose(v, 0)] = -1
        rs[np.isclose(v, 0)] = 1

        return ls, rs


class IsPos(G):
    def __init__(self, weight, scale, shift):
        super().__init__(weight, scale, shift)
        self._is_convex = True

    def evaluate_raw(self, v):
        v = np.asarray(v)
        return np.where(v >= 0, 0, np.inf)

    def prox_raw(self, rho, v):
        v = np.asarray(v)
        return np.where(v < 0, 0, v)

    def subdiff_raw(self, v):
        v = np.asarray(v)
        ls = np.zeros(v.shape)
        rs = np.zeros(v.shape)

        ls[v < 0] = np.nan
        rs[v < 0] = np.nan

        ls[v == 0] = -np.inf
        rs[v == 0] = 0

        return ls, rs


class IsNeg(G):
    def __init__(self, weight, scale, shift):
        super().__init__(weight, scale, shift)
        self._is_convex = True

    def evaluate_raw(self, v):
        v = np.asarray(v)
        return np.where(v <= 0, 0, np.inf)

    def prox_raw(self, rho, v):
        v = np.asarray(v)
        return np.where(v > 0, 0, v)

    def subdiff_raw(self, v):
        v = np.asarray(v)
        ls = np.zeros(v.shape)
        rs = np.zeros(v.shape)

        ls[v > 0] = np.nan
        rs[v > 0] = np.nan

        ls[v == 0] = 0
        rs[v == 0] = np.inf

        return ls, rs


class IsBound(G):
    def __init__(self, weight, scale, shift, lb, ub):
        super().__init__(weight, scale, shift)
        self._lb = lb
        self._ub = ub
        self._is_convex = True

    def evaluate_raw(self, v):
        return np.where((v >= self._lb) & (v <= self._ub), 0, np.inf)

    def prox_raw(self, rho, v):
        output = np.where(v >= self._ub, self._ub, v)
        output = np.where(output <= self._lb, self._lb, output)
        return output

    def subdiff_raw(self, v):
        v = np.asarray(v)
        ls = np.zeros(v.shape)
        rs = np.zeros(v.shape)

        ls[v > self._ub] = np.nan
        rs[v > self._ub] = np.nan

        ls[v == self._ub] = 0
        rs[v == self._ub] = np.inf

        ls[v < self._lb] = np.nan
        rs[v < self._lb] = np.nan

        ls[v == self._lb] = -np.inf
        rs[v == self._lb] = 0

        return ls, rs


class IsZero(G):
    def __init__(self, weight, scale, shift):
        super().__init__(weight, scale, shift)
        self._is_convex = True

    def evaluate_raw(self, v):
        return np.where(v == 0, 0, np.inf)

    def prox_raw(self, rho, v):
        return np.zeros(np.asarray(v).shape)

    def subdiff_raw(self, v):
        v = np.asarray(v)
        ls = np.nan * np.ones(v.shape)
        rs = np.nan * np.ones(v.shape)

        ls[v == 0] = -np.inf
        rs[v == 0] = np.inf

        return ls, rs


class Pos(G):
    def __init__(self, weight, scale, shift):
        super().__init__(weight, scale, shift)
        self._is_convex = True

    def evaluate_raw(self, v):
        return np.maximum(v, 0)

    def prox_raw(self, rho, v):
        output = np.where(v <= 0, v, 0)
        output = np.where(v > 1 / rho, v - 1 / rho, output)
        return output

    def subdiff_raw(self, v):
        v = np.asarray(v)
        ls = np.zeros(v.shape)
        rs = np.zeros(v.shape)

        ls[v > 0] = 1
        rs[v > 0] = 1

        # TODO: change this to is_close?
        ls[v == 0] = 0
        rs[v == 0] = 1

        return ls, rs


class Neg(G):
    def __init__(self, weight, scale, shift):
        super().__init__(weight, scale, shift)
        self._is_convex = True

    def evaluate_raw(self, v):
        return np.maximum(-v, 0)

    def prox_raw(self, rho, v):
        return np.where(v < -1 / rho, v + 1 / rho, v)

    def subdiff_raw(self, v):
        v = np.asarray(v)
        ls = np.zeros(v.shape)
        rs = np.zeros(v.shape)

        ls[v < 0] = -1
        rs[v < 0] = -1

        # TODO: change this to is_close?
        ls[v == 0] = -1
        rs[v == 0] = 0

        return ls, rs


class Card(G):
    def __init__(self, weight, scale, shift):
        super().__init__(weight, scale, shift)
        self._is_convex = False

    def evaluate_raw(self, v):
        # TODO: change this to isclose?
        return np.where(v == 0, 0, 1)

    def prox_raw(self, rho, v):
        # print(np.sqrt(2/np.max(rho)), np.max(np.abs(v)))
        return np.where(np.abs(v) < np.sqrt(2 / rho), 0, v)

    def subdiff_raw(self, v):
        v = np.asarray(v)
        ls = np.nan * np.ones(v.shape)
        rs = np.nan * np.ones(v.shape)

        ls[v == 0] = 0
        rs[v == 0] = 0

        return ls, rs


class CardConstr(G):
    def __init__(self, weight, scale, shift, k):
        super().__init__(weight, scale, shift)
        self._k = k
        self._is_convex = False

    def evaluate_raw(self, v):
        constr_satisfied = np.count_nonzero(v) <= self._k
        if constr_satisfied:
            return np.zeros(v.shape)
        else:
            return np.inf * np.ones(v.shape)

    def prox_raw(self, rho, v):
        non_top_k = np.argpartition(v, -self._k)[:-self._k]
        v[non_top_k] = 0
        return v
    
    def subdiff_raw(self, v):
        return np.nan * np.ones(v.shape), np.nan * np.ones(v.shape)


class Quantile(G):
    def __init__(self, weight, scale, shift, tau):
        super().__init__(weight, scale, shift)
        self._tau = tau
        self._is_convex = True

    def evaluate_raw(self, v):
        return 0.5 * np.abs(v) + (self._tau - 0.5) * v

    def prox_raw(self, rho, v):
        v_mod = np.asarray(v) + 1 / rho * (0.5 - self._tau)
        return np.where(
            np.abs(v_mod) <= 1 / (2 * rho), 0, v_mod - np.sign(v_mod) * 1 / (2 * rho)
        )

    def subdiff_raw(self, v):
        v = np.asarray(v)
        ls = np.zeros(v.shape)
        rs = np.zeros(v.shape)

        ls[v > 0] = self._tau
        rs[v > 0] = self._tau

        ls[v < 0] = self._tau - 1
        rs[v < 0] = self._tau - 1

        # TODO: change to is_close?
        ls[v == 0] = self._tau - 1
        rs[v == 0] = self._tau

        return ls, rs


class Huber(G):
    def __init__(self, weight, scale, shift, M):
        super().__init__(weight, scale, shift)
        self._M = M
        self._is_convex = True

    def evaluate_raw(self, v):
        abs_v = np.abs(v)
        return np.where(
            abs_v <= self._M, abs_v**2, 2 * self._M * abs_v - self._M * self._M
        )

    def prox_raw(self, rho, v):
        return np.where(
            np.abs(v) <= self._M * (rho + 2) / rho,
            rho / (2 + rho) * v,
            v - np.sign(v) * 2 * self._M / rho,
        )

    def subdiff_raw(self, v):
        v = np.asarray(v)
        ls = np.zeros(v.shape)
        rs = np.zeros(v.shape)

        abs_v = np.abs(v)

        ls[abs_v <= self._M] = 2 * v[abs_v <= self._M]
        rs[abs_v <= self._M] = 2 * v[abs_v <= self._M]

        ls[v > self._M] = 2 * self._M
        rs[v > self._M] = 2 * self._M

        ls[v < self._M] = -2 * self._M
        rs[v < self._M] = -2 * self._M

        return ls, rs


class IsInt(G):
    def __init__(self, weight, scale, shift):
        super().__init__(weight, scale, shift)
        self._is_convex = False

    def evaluate_raw(self, v):
        return np.where(
            np.isclose(np.mod(v, 1), 0) | np.isclose(np.mod(v, 1), 1), 0, np.inf
        )
        # TODO: change this to something like np.isclose(v, np.rint(v))

    def prox_raw(self, rho, v):
        return np.rint(v)

    def subdiff_raw(self, v):
        v = np.asarray(v)
        ls = np.nan * np.ones(v.shape)
        rs = np.nan * np.ones(v.shape)

        int_indices = self.evaluate_raw(v)

        ls[int_indices == 0] = 0
        rs[int_indices == 0] = 0

        return ls, rs


class IsFiniteSet(G):
    def __init__(self, weight, scale, shift, S):
        super().__init__(weight, scale, shift)
        self._S = np.array(list(S))
        self._is_convex = False

    def evaluate_raw(self, v):
        v = np.asarray(v)
        is_almost_in_S = np.isclose(v.reshape((-1, 1)), self._S.reshape((1, -1))).any(
            axis=1
        )
        return np.where(is_almost_in_S, 0, np.inf)

    def prox_raw(self, rho, v):
        v = np.asarray(v)
        diffs = np.subtract(v.reshape((-1, 1)), self._S.reshape((1, -1)))
        idx = np.argmin(np.abs(diffs), axis=1)
        return self._S[idx]

    def subdiff_raw(self, v):
        v = np.asarray(v)
        ls = np.nan * np.ones(v.shape)
        rs = np.nan * np.ones(v.shape)

        in_set_indices = self.evaluate_raw(v)

        ls[in_set_indices == 0] = 0
        rs[in_set_indices == 0] = 0

        return ls, rs


class IsBool(G):
    def __init__(self, weight, scale, shift):
        super().__init__(weight, scale, shift)
        self._S = np.array([0, 1])
        self._is_convex = False

    def evaluate_raw(self, v):
        v = np.asarray(v)
        is_almost_in_S = np.isclose(v.reshape((-1, 1)), self._S.reshape((1, -1))).any(
            axis=1
        )
        return np.where(is_almost_in_S, 0, np.inf)

    def prox_raw(self, rho, v):
        v = np.asarray(v)
        diffs = np.subtract(v.reshape((-1, 1)), self._S.reshape((1, -1)))
        idx = np.argmin(np.abs(diffs), axis=1)
        return self._S[idx]

    def subdiff_raw(self, v):
        v = np.asarray(v)
        ls = np.nan * np.ones(v.shape)
        rs = np.nan * np.ones(v.shape)

        is_bool_indices = self.evaluate_raw(v)

        ls[is_bool_indices == 0] = 0
        rs[is_bool_indices == 0] = 0

        return ls, rs


class GCollection:
    def __init__(self, g_list, dim, relax=False):
        self._g_list = []
        self._is_convex = True
        self._all_zeros = True
        self._full_g = False
        self.dim = dim
        self.bool_ranges = np.vstack(
            [np.full((len(g_list), dim), False), np.full((1, dim), True)]
        )

        for i, g in enumerate(g_list):
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
            range = g["range"]
            name = g["g"]

            if name == "zero":
                func = Zero(weight, scale, shift)
            elif name == "abs":
                func = Abs(weight, scale, shift)
            elif name == "is_pos":
                func = IsPos(weight, scale, shift)
            elif name == "is_neg":
                func = IsNeg(weight, scale, shift)
            elif name == "is_bound":
                if "args" in g and "lb" in g["args"]:
                    lb = g["args"]["lb"]
                else:
                    lb = 0
                if "args" in g and "ub" in g["args"]:
                    ub = g["args"]["ub"]
                else:
                    ub = 1
                func = IsBound(weight, scale, shift, lb, ub)
            elif name == "is_zero":
                func = IsZero(weight, scale, shift)
            elif name == "pos":
                func = Pos(weight, scale, shift)
            elif name == "neg":
                func = Neg(weight, scale, shift)
            elif name == "card":
                if not relax:
                    func = Card(weight, scale, shift)
                else:
                    func = Abs(weight, scale, shift)
            elif name == "card_constr":
                if not relax:
                    if "args" in g and "k" in g["args"]:
                        k = g["args"]["k"]
                    else:
                        k = 1
                    func = CardConstr(weight, scale, shift, k)
                else:
                    # TODO: check this relaxation
                    func = Abs(1e-1, scale, shift)
            elif name == "quantile":
                if "args" in g and "tau" in g["args"]:
                    tau = g["args"]["tau"]
                else:
                    tau = 0.5
                func = Quantile(weight, scale, shift, tau)
            elif name == "huber":
                if "args" in g and "M" in g["args"]:
                    M = g["args"]["M"]
                    if M <= 0:
                        raise ValueError("Huber parameter M must be > 0.")
                else:
                    M = 1
                func = Huber(weight, scale, shift, M)
            elif name == "is_int":
                if not relax:
                    func = IsInt(weight, scale, shift)
                else:
                    func = Zero(weight, scale, shift)
            elif name == "is_finite_set":
                if "args" in g and "S" in g["args"]:
                    value_set = g["args"]["S"]
                else:
                    raise ValueError("is_finite_set set must be specified.")
                if not relax:
                    func = IsFiniteSet(weight, scale, shift, value_set)
                else:
                    func = IsBound(weight, scale, shift, min(value_set), max(value_set))
            elif name == "is_bool":
                if not relax:
                    func = IsBool(weight, scale, shift)
                else:
                    func = IsBound(weight, scale, shift, 0, 1)

            if not func._is_convex:
                self._is_convex = False
            self._g_list.append({"range": range, "func": func})

            if name != "zero":
                self._all_zeros = False
                self.bool_ranges[-1, range[0] : range[1]] = False
                self.bool_ranges[i, range[0] : range[1]] = True
            
        # Check if user-specified g's take up the whole dim
        if np.all(~self.bool_ranges[-1]):
            self._full_g = True
            self.bool_ranges = self.bool_ranges[:-1]

    def evaluate(self, v):
        output = np.zeros(np.asarray(v).shape)

        for item in self._g_list:
            start_index, end_index = item["range"]
            func = item["func"]
            output[start_index:end_index] = func.evaluate(v[start_index:end_index])

        return np.sum(output)

    def prox(self, rho_vec, equil_scaling, v):
        output = np.copy(v)

        for item in self._g_list:
            start_index, end_index = item["range"]
            func = item["func"]
            output[start_index:end_index] = func.prox(
                rho_vec[start_index:end_index],
                equil_scaling[start_index:end_index],
                v[start_index:end_index],
            )

        return output

    def subdiff(self, equil_scaling, obj_scale, v):
        v = np.asarray(v)
        ls = np.zeros(v.shape)
        rs = np.zeros(v.shape)

        for item in self._g_list:
            start_index, end_index = item["range"]
            func = item["func"]
            g_ls, g_rs = func.subdiff(
                equil_scaling[start_index:end_index],
                obj_scale,
                v[start_index:end_index],
            )
            ls[start_index:end_index] = g_ls
            rs[start_index:end_index] = g_rs

        return ls, rs
