import numpy as np
import scipy as sp
import cvxpy as cp
import qss
from qss import proximal


class TestZero:
    v = np.arange(5)
    g = proximal.Zero(weight=1, scale=1, shift=0)

    def test_g(self):
        res = self.g.evaluate(self.v)
        assert np.all(res == np.zeros(len(self.v)))

    def test_prox(self):
        res = self.g.prox(10, np.ones(self.v.shape), self.v)
        assert np.all(self.v == res)

    def test_subdiff(self):
        ls, rs = self.g.subdiff(np.ones(self.v.shape), 1, self.v)
        res = ls == rs
        assert np.all(res == 0)


class TestAbs:
    v1 = np.array([-2, -1, -1e-30, 0, 1e-30, 1, 2])
    g = proximal.Abs(weight=1, scale=1, shift=0)

    def test_g(self):
        assert np.all(np.abs(self.v1) == self.g.evaluate(self.v1))

    def test_prox(self):
        rho = 10
        res_qss = self.g.prox(rho, np.ones(self.v1.shape), self.v1)
        x = cp.Variable(len(self.v1))
        objective = cp.Minimize(cp.norm(x, 1) + rho / 2 * cp.norm(x - self.v1) ** 2)
        cp.Problem(objective).solve()
        assert np.all(np.isclose(res_qss, x.value))

    def test_subdiff(self):
        ls, rs = self.g.subdiff(np.ones(self.v1.shape), 1, self.v1)
        assert np.all(ls == np.array([-1, -1, -1, -1, -1, 1, 1]))
        assert np.all(rs == np.array([-1, -1, 1, 1, 1, 1, 1]))


class TestIsPos:
    v1 = np.array([-1, -1e-30, 0, 1e-30, 1])
    g = proximal.IsPos(weight=1, scale=1, shift=0)

    def test_g(self):
        res = self.g.evaluate(self.v1)
        assert np.all(res == np.array([np.inf, np.inf, 0, 0, 0]))

    def test_prox(self):
        rho = 10
        res = self.g.prox(rho, np.ones_like(self.v1), self.v1)
        assert np.all(res == np.array([0, 0, 0, 1e-30, 1]))

    def test_subdiff(self):
        rho = 10
        ls, rs = self.g.subdiff(np.ones_like(self.v1), 1, self.v1)
        # Using allclose because a == b doesn't handle NaNs
        assert np.allclose(
            ls, np.array([np.nan, np.nan, -np.inf, 0, 0]), equal_nan=True
        )
        assert np.allclose(rs, np.array([np.nan, np.nan, 0, 0, 0]), equal_nan=True)


class TestIsNeg:
    v1 = np.array([-1, -1e-30, 0, 1e-30, 1])
    g = proximal.IsNeg(weight=1, scale=1, shift=0)

    def test_g(self):
        res = self.g.evaluate(self.v1)
        assert np.all(res == np.array([0, 0, 0, np.inf, np.inf]))

    def test_prox(self):
        rho = 10
        res = self.g.prox(rho, np.ones_like(self.v1), self.v1)
        assert np.all(res == np.array([-1, -1e-30, 0, 0, 0]))

    def test_subdiff(self):
        rho = 10
        ls, rs = self.g.subdiff(np.ones_like(self.v1), 1, self.v1)
        assert np.allclose(ls, np.array([0, 0, 0, np.nan, np.nan]), equal_nan=True)
        assert np.allclose(rs, np.array([0, 0, np.inf, np.nan, np.nan]), equal_nan=True)


class TestIsBound:
    v1 = np.array([-2, -1, 0, 1, 2])

    def test_g(self):
        g = proximal.IsBound(weight=1, scale=1, shift=0, lb=0, ub=1)
        res = g.evaluate(self.v1)
        assert np.all(res == np.array([np.inf, np.inf, 0, 0, np.inf]))

        g = proximal.IsBound(weight=1, scale=1, shift=0, lb=-1, ub=0)
        res = g.evaluate(self.v1)
        assert np.all(res == np.array([np.inf, 0, 0, np.inf, np.inf]))

    def test_prox(self):
        g = proximal.IsBound(weight=1, scale=1, shift=0, lb=0, ub=1)
        rho = 10
        res = g.prox(rho, np.ones_like(self.v1), self.v1)
        assert np.all(res == np.array([0, 0, 0, 1, 1]))

        g = proximal.IsBound(weight=1, scale=1, shift=0, lb=-1, ub=0)
        res = g.prox(rho, np.ones_like(self.v1), self.v1)
        assert np.all(res == np.array([-1, -1, 0, 0, 0]))

    def test_subdiff(self):
        g = proximal.IsBound(weight=1, scale=1, shift=0, lb=0, ub=1)
        ls, rs = g.subdiff(np.ones_like(self.v1), 1, self.v1)
        assert np.allclose(
            ls, np.array([np.nan, np.nan, -np.inf, 0, np.nan]), equal_nan=True
        )
        assert np.allclose(
            rs, np.array([np.nan, np.nan, 0, np.inf, np.nan]), equal_nan=True
        )

        g = proximal.IsBound(weight=1, scale=1, shift=0, lb=-1, ub=0)
        ls, rs = g.subdiff(np.ones_like(self.v1), 1, self.v1)
        assert np.allclose(
            ls, np.array([np.nan, -np.inf, 0, np.nan, np.nan]), equal_nan=True
        )
        assert np.allclose(
            rs, np.array([np.nan, 0, np.inf, np.nan, np.nan]), equal_nan=True
        )


class TestIsZero:
    v1 = np.array([1, 0, -0.001])
    g = proximal.IsZero(weight=1, scale=1, shift=0)

    def test_g(self):
        res = self.g.evaluate(self.v1)
        assert np.all(res == np.array([np.inf, 0, np.inf]))

    def test_prox(self):
        rho = 10
        res = self.g.prox(rho, np.ones_like(self.v1), self.v1)
        assert np.all(res == np.array([0, 0, 0]))

    def test_subdiff(self):
        ls, rs = self.g.subdiff(np.ones_like(self.v1), 1, self.v1)
        assert np.allclose(ls, np.array([np.nan, -np.inf, np.nan]), equal_nan=True)
        assert np.allclose(rs, np.array([np.nan, np.inf, np.nan]), equal_nan=True)


class TestPos:
    v1 = np.array([-1, -1e-30, 0, 1e-30, 1])
    g = proximal.Pos(weight=1, scale=1, shift=0)

    def test_g(self):
        res = self.g.evaluate(self.v1)
        assert np.all(res == np.array([0, 0, 0, 1e-30, 1]))

    def test_prox(self):
        rho = 10
        res_qss = self.g.prox(rho, np.ones_like(self.v1), self.v1)
        x = cp.Variable(len(self.v1))
        objective = cp.Minimize(cp.sum(cp.pos(x)) + rho / 2 * cp.norm(x - self.v1) ** 2)
        # Use SCS as MOSEK gives slightly different answer
        cp.Problem(objective).solve(solver=cp.SCS)
        assert np.allclose(res_qss, x.value, atol=1e-4)

    def test_subdiff(self):
        ls, rs = self.g.subdiff(np.ones_like(self.v1), 1, self.v1)
        assert np.all(ls == np.array([0, 0, 0, 1, 1]))
        assert np.all(rs == np.array([0, 0, 1, 1, 1]))


class TestNeg:
    v1 = np.array([-1, -1e-30, 0, 1e-30, 1])
    g = proximal.Neg(weight=1, scale=1, shift=0)

    def test_g(self):
        res = self.g.evaluate(self.v1)
        assert np.all(res == np.array([1, 1e-30, 0, 0, 0]))

    def test_prox(self):
        rho = 10
        res_qss = self.g.prox(rho, np.ones_like(self.v1), self.v1)
        x = cp.Variable(len(self.v1))
        objective = cp.Minimize(cp.sum(cp.neg(x)) + rho / 2 * cp.norm(x - self.v1) ** 2)
        cp.Problem(objective).solve(solver=cp.SCS)
        assert np.allclose(res_qss, x.value, atol=1e-4)

    def test_subdiff(self):
        ls, rs = self.g.subdiff(np.ones_like(self.v1), 1, self.v1)
        assert np.all(ls == np.array([-1, -1, -1, 0, 0]))
        assert np.all(rs == np.array([-1, -1, 0, 0, 0]))


class TestCard:
    v1 = np.array([-1, -1e-30, 0, 1e-30, 1])
    g = proximal.Card(weight=1, scale=1, shift=0)

    def test_g(self):
        res = self.g.evaluate(self.v1)
        assert np.all(res == np.array([1, 1, 0, 1, 1]))

    def test_prox(self):
        rho = 10
        res = self.g.prox(rho, np.ones_like(self.v1), self.v1)
        assert np.all(res == np.array([-1, 0, 0, 0, 1]))

    def test_subdiff(self):
        ls, rs = self.g.subdiff(np.ones_like(self.v1), 1, self.v1)
        assert np.allclose(
            ls, np.array([np.nan, np.nan, 0, np.nan, np.nan]), equal_nan=True
        )
        assert np.allclose(
            rs, np.array([np.nan, np.nan, 0, np.nan, np.nan]), equal_nan=True
        )


class TestGCollection:
    g1 = [{"g": "abs", "range": (2, 3)}, {"g": "zero", "range": (3, 5)}]
    gcoll1 = proximal.GCollection(g1, 5)

    g2 = []
    gcoll2 = proximal.GCollection(g2, 10)

    g3 = [{"g": "zero", "range": (10, 20)}]
    gcoll3 = proximal.GCollection(g3, 25)

    def test_flags(self):
        assert self.gcoll1._is_convex is True
        assert self.gcoll1._all_zeros is False

        assert self.gcoll2._is_convex is True
        assert self.gcoll2._all_zeros is True

        assert self.gcoll3._is_convex is True
        assert self.gcoll3._all_zeros is True
