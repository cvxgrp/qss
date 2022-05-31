import numpy as np
import scipy as sp
import qss
import pytest


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


class TestConstraints:
    data = {}
    data["P"] = sp.sparse.eye(100)
    data["q"] = np.ones(100)
    data["r"] = 10
    data["g"] = [{"g": "is_pos", "range": (10, 20)}, {"g": "huber", "range": (30, 50)}]

    def test_no_b(self):
        self.data["A"] = sp.sparse.csc_matrix(sp.sparse.eye(100))[:50, :]
        with pytest.raises(ValueError) as exc_info:
            qss.QSS(self.data)
        print(exc_info.value)

    def test_no_A(self):
        self.data["A"] = None
        self.data["b"] = np.ones(50)
        with pytest.raises(ValueError) as exc_info:
            qss.QSS(self.data)
        print(exc_info.value)
