import numpy as np
import scipy as sp
import qss
import pytest


class TestConstraints:
    dim = 100
    constr_dim = 50
    data = {}
    data["P"] = sp.sparse.eye(dim)
    data["q"] = np.ones(dim)
    data["r"] = 10
    data["g"] = [{"g": "is_pos", "range": (10, 20)}, {"g": "huber", "range": (30, 50)}]

    def test_no_b(self):
        self.data["A"] = sp.sparse.csc_matrix(sp.sparse.eye(self.dim))[
            : self.constr_dim, :
        ]
        with pytest.raises(ValueError) as exc_info:
            qss.QSS(self.data)
        print(exc_info.value)

    def test_no_A(self):
        self.data["A"] = None
        self.data["b"] = np.ones(self.constr_dim)
        with pytest.raises(ValueError) as exc_info:
            qss.QSS(self.data)
        print(exc_info.value)

    def test_A_bad_shape(self):
        self.data["A"] = sp.sparse.csc_matrix(sp.sparse.eye(self.dim - 1))[
            : self.constr_dim, :
        ]
        self.data["b"] = np.ones(self.constr_dim)
        with pytest.raises(ValueError) as exc_info:
            qss.QSS(self.data)
        print(exc_info.value)

    def test_A_b_dimension_mismatch(self):
        self.data["A"] = sp.sparse.csc_matrix(sp.sparse.eye(self.dim))[
            : self.constr_dim, :
        ]
        self.data["b"] = np.ones(self.constr_dim + 1)
        with pytest.raises(ValueError) as exc_info:
            qss.QSS(self.data)
        print(exc_info.value)


class Testg:
    data = {}
    data["P"] = sp.sparse.eye(100)
    data["q"] = np.ones(100)
    data["r"] = 10
    data["A"] = sp.sparse.csc_matrix(sp.sparse.eye(100))[:50, :]
    data["b"] = np.ones(50)

    def test_bad_name(self):
        self.data["g"] = [{"g": "bad_name", "range": (10, 20)}]
        with pytest.raises(ValueError) as exc_info:
            qss.QSS(self.data)
        print(exc_info.value)

    def test_no_range(self):
        self.data["g"] = [{"g": "abs"}]
        with pytest.raises(ValueError) as exc_info:
            qss.QSS(self.data)
        print(exc_info.value)

    def test_bad_range(self):
        self.data["g"] = [{"g": "abs", "range": (-10, 20)}]
        with pytest.raises(ValueError) as exc_info:
            qss.QSS(self.data)
        print(exc_info.value)

        self.data["g"] = [{"g": "abs", "range": (0, 120)}]
        with pytest.raises(ValueError) as exc_info:
            qss.QSS(self.data)
        print(exc_info.value)

        self.data["g"] = [{"g": "abs", "range": (20, 10)}]
        with pytest.raises(ValueError) as exc_info:
            qss.QSS(self.data)
        print(exc_info.value)

    def test_overlap(self):
        self.data["g"] = [
            {"g": "abs", "range": (0, 10)},
            {"g": "abs", "range": (10, 20)},
            {"g": "abs", "range": (35, 45)},
            {"g": "abs", "range": (30, 35)},
        ]
        qss.QSS(self.data)

        self.data["g"] = [
            {"g": "abs", "range": (0, 10)},
            {"g": "abs", "range": (10, 20)},
            {"g": "abs", "range": (35, 45)},
            {"g": "abs", "range": (30, 36)},
        ]
        with pytest.raises(ValueError) as exc_info:
            qss.QSS(self.data)
        print(exc_info.value)


class TestOptions:
    data = {}
    data["P"] = sp.sparse.eye(100)
    data["q"] = np.ones(100)
    data["r"] = 10
    data["A"] = sp.sparse.csc_matrix(sp.sparse.eye(100))[:50, :]
    data["b"] = np.ones(50)
    data["g"] = [{"g": "is_pos", "range": (10, 20)}, {"g": "huber", "range": (30, 50)}]
    solver = qss.QSS(data)

    def test_bad_max_iter(self):
        with pytest.raises(ValueError) as exc_info:
            self.solver.solve(max_iter=100.0)
        print(exc_info.value)

        with pytest.raises(ValueError) as exc_info:
            self.solver.solve(max_iter="100")
        print(exc_info.value)
