import numpy as np
import scipy as sp
import cvxpy as cp
import qss
from qss import proximal


class TestZero:
    v = np.arange(5)

    def test_g(self):
        res = proximal.g_zero(self.v, {})
        assert np.all(res == np.zeros(len(self.v)))

    def test_prox(self):
        res = proximal.prox_zero(10, self.v, {})
        assert np.all(self.v == res)

    def test_subdiff(self):
        ls, rs = proximal.subdiff_zero(self.v, {})
        res = ls == rs
        return np.all(res == 0)


class TestAbs:
    v1 = np.array([-2, -1, -1e-30, 0, 1e-30, 1, 2])

    def test_g(self):
        assert np.all(np.abs(self.v1) == proximal.g_abs(self.v1, {}))

    def test_prox(self):
        rho = 10
        res_qss = proximal.prox_abs(rho, self.v1, {})
        x = cp.Variable(len(self.v1))
        objective = cp.Minimize(cp.norm(x, 1) + rho / 2 * cp.norm(x - self.v1) ** 2)
        cp.Problem(objective).solve()
        assert np.all(np.isclose(res_qss, x.value))

    def test_subdiff(self):
        ls, rs = proximal.subdiff_abs(self.v1, {})
        assert np.all(ls == np.array([-1, -1, -1, -1, -1, 1, 1]))
        assert np.all(rs == np.array([-1, -1, 1, 1, 1, 1, 1]))


class TestIsPos:
    v1 = np.array([-1, -1e-30, 0, 1e-30, 1])

    def test_g(self):
        res = proximal.g_is_pos(self.v1, {})
        assert np.all(res == np.array([np.inf, np.inf, 0, 0, 0]))

    def test_prox(self):
        rho = 10
        res = proximal.prox_is_pos(rho, self.v1, {})
        assert np.all(res == np.array([0, 0, 0, 1e-30, 1]))

    def test_subdiff(self):
        rho = 10
        ls, rs = proximal.subdiff_is_pos(self.v1, {})
        # Using allclose because a == b doesn't handle NaNs
        assert np.allclose(
            ls, np.array([np.nan, np.nan, -np.inf, 0, 0]), equal_nan=True
        )
        assert np.allclose(rs, np.array([np.nan, np.nan, 0, 0, 0]), equal_nan=True)


class TestIsNeg:
    v1 = np.array([-1, -1e-30, 0, 1e-30, 1])

    def test_g(self):
        res = proximal.g_is_neg(self.v1, {})
        assert np.all(res == np.array([0, 0, 0, np.inf, np.inf]))

    def test_prox(self):
        rho = 10
        res = proximal.prox_is_neg(rho, self.v1, {})
        assert np.all(res == np.array([-1, -1e-30, 0, 0, 0]))

    def test_subdiff(self):
        rho = 10
        ls, rs = proximal.subdiff_is_neg(self.v1, {})
        assert np.allclose(ls, np.array([0, 0, 0, np.nan, np.nan]), equal_nan=True)
        assert np.allclose(rs, np.array([0, 0, np.inf, np.nan, np.nan]), equal_nan=True)


class TestIsBound:
    v1 = np.array([-2, -1, 0, 1, 2])

    def test_g(self):
        res = proximal.g_is_bound(self.v1, {})
        assert np.all(res == np.array([np.inf, np.inf, 0, 0, np.inf]))

        res = proximal.g_is_bound(self.v1, {"lb": -1, "ub": 0})
        assert np.all(res == np.array([np.inf, 0, 0, np.inf, np.inf]))

    def test_prox(self):
        rho = 10
        res = proximal.prox_is_bound(rho, self.v1, {})
        assert np.all(res == np.array([0, 0, 0, 1, 1]))

        res = proximal.prox_is_bound(rho, self.v1, {"lb": -1, "ub": 0})
        assert np.all(res == np.array([-1, -1, 0, 0, 0]))

    def test_subdiff(self):
        ls, rs = proximal.subdiff_is_bound(self.v1, {})
        assert np.allclose(
            ls, np.array([np.nan, np.nan, -np.inf, 0, np.nan]), equal_nan=True
        )
        assert np.allclose(
            rs, np.array([np.nan, np.nan, 0, np.inf, np.nan]), equal_nan=True
        )

        ls, rs = proximal.subdiff_is_bound(self.v1, {"lb": -1, "ub": 0})
        assert np.allclose(
            ls, np.array([np.nan, -np.inf, 0, np.nan, np.nan]), equal_nan=True
        )
        assert np.allclose(
            rs, np.array([np.nan, 0, np.inf, np.nan, np.nan]), equal_nan=True
        )


class TestIsZero:
    v1 = np.array([1, 0, -0.001])

    def test_g(self):
        res = proximal.g_is_zero(self.v1, {})
        assert np.all(res == np.array([np.inf, 0, np.inf]))

    def test_prox(self):
        rho = 10
        res = proximal.prox_is_zero(rho, self.v1, {})
        assert np.all(res == np.array([0, 0, 0]))

    def test_subdiff(self):
        ls, rs = proximal.subdiff_is_zero(self.v1, {})
        assert np.allclose(ls, np.array([np.nan, -np.inf, np.nan]), equal_nan=True)
        assert np.allclose(rs, np.array([np.nan, np.inf, np.nan]), equal_nan=True)


class TestPos:
    v1 = np.array([-1, -1e-30, 0, 1e-30, 1])

    def test_g(self):
        res = proximal.g_pos(self.v1, {})
        assert np.all(res == np.array([0, 0, 0, 1e-30, 1]))

    def test_prox(self):
        rho = 10
        res_qss = proximal.prox_pos(rho, self.v1, {})
        x = cp.Variable(len(self.v1))
        objective = cp.Minimize(cp.sum(cp.pos(x)) + rho / 2 * cp.norm(x - self.v1) ** 2)
        # Use SCS as MOSEK gives slightly different answer
        cp.Problem(objective).solve(solver=cp.SCS)
        assert np.allclose(res_qss, x.value)

    def test_subdiff(self):
        ls, rs = proximal.subdiff_pos(self.v1, {})
        assert np.all(ls == np.array([0, 0, 0, 1, 1]))
        assert np.all(rs == np.array([0, 0, 1, 1, 1]))


class TestNeg:
    v1 = np.array([-1, -1e-30, 0, 1e-30, 1])

    def test_g(self):
        res = proximal.g_neg(self.v1, {})
        assert np.all(res == np.array([1, 1e-30, 0, 0, 0]))

    def test_prox(self):
        rho = 10
        res_qss = proximal.prox_neg(rho, self.v1, {})
        x = cp.Variable(len(self.v1))
        objective = cp.Minimize(cp.sum(cp.neg(x)) + rho / 2 * cp.norm(x - self.v1) ** 2)
        cp.Problem(objective).solve(solver=cp.SCS)
        assert np.allclose(res_qss, x.value)

    def test_subdiff(self):
        ls, rs = proximal.subdiff_neg(self.v1, {})
        assert np.all(ls == np.array([-1, -1, -1, 0, 0]))
        assert np.all(rs == np.array([-1, -1, 0, 0, 0]))


class TestCard:
    v1 = np.array([-1, -1e-30, 0, 1e-30, 1])

    def test_g(self):
        res = proximal.g_card(self.v1, {})
        assert np.all(res == np.array([1, 1, 0, 1, 1]))

    def test_prox(self):
        rho = 10
        res = proximal.prox_card(rho, self.v1, {})
        assert np.all(res == np.array([-1, 0, 0, 0, 1]))

    def test_subdiff(self):
        ls, rs = proximal.subdiff_card(self.v1, {})
        print(ls, rs)
        assert np.allclose(ls, np.array([np.nan, np.nan, 0, np.nan, np.nan]), equal_nan=True)
        assert np.allclose(rs, np.array([np.nan, np.nan, 0, np.nan, np.nan]), equal_nan=True)