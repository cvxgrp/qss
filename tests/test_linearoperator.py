import numpy as np
import scipy as sp
import pytest
import qss
from qss.linearoperator import LinearOperator

np.random.seed(1234)


class TestConstructor:
    I = sp.sparse.eye(10)
    A = sp.sparse.rand(5, 10, density=0.9)
    P = sp.sparse.rand(10, 10, density=0.9)
    P = P.T @ P

    def mv(v):
        return 0.1 * v

    def rmv(v):
        return 0.1 * v

    F = sp.sparse.linalg.LinearOperator(
        (A.shape[0], A.shape[0]), matvec=mv, rmatvec=rmv
    )

    def test_list_of_lists(self):
        with pytest.raises(ValueError) as exc_info:
            LinearOperator(self.A)
        print(exc_info.value)

        with pytest.raises(ValueError) as exc_info:
            LinearOperator([self.A])
        print(exc_info.value)

        with pytest.raises(ValueError) as exc_info:
            LinearOperator([self.A, self.A])
        print(exc_info.value)

    def test_dimension_mismatch(self):
        with pytest.raises(ValueError) as exc_info:
            LinearOperator([[self.A, self.P, None]])
        print(exc_info.value)

        with pytest.raises(ValueError) as exc_info:
            LinearOperator([[None, self.A, self.P]])
        print(exc_info.value)

        with pytest.raises(ValueError) as exc_info:
            LinearOperator([[self.A.T], [self.P]])
        print(exc_info.value)

        with pytest.raises(ValueError) as exc_info:
            LinearOperator([[self.A], [self.F]])
        print(exc_info.value)

        with pytest.raises(ValueError) as exc_info:
            LinearOperator([[self.A, self.A], [self.A]])
        print(exc_info.value)

    def test_all_nones(self):
        with pytest.raises(ValueError) as exc_info:
            LinearOperator([[None, None]])
        print(exc_info.value)

        with pytest.raises(ValueError) as exc_info:
            LinearOperator([[None, None], [self.A, self.A]])
        print(exc_info.value)

        with pytest.raises(ValueError) as exc_info:
            LinearOperator([[None, self.A], [None, self.A]])
        print(exc_info.value)

    def test_shape(self):
        linop = LinearOperator([[self.P, self.A.T], [self.A, self.F]])
        assert linop.shape[0] == self.P.shape[0] + self.A.shape[0]
        assert linop.shape[1] == self.P.shape[1] + self.A.shape[0]


class TestMatvec:
    I = sp.sparse.eye(10)
    A = sp.sparse.rand(5, 10, density=0.9)
    P = sp.sparse.rand(10, 10, density=0.9)
    P = P.T @ P

    def mv(v):
        return 0.1 * v

    def rmv(v):
        return 0.1 * v

    F = sp.sparse.linalg.LinearOperator(
        (A.shape[0], A.shape[0]), matvec=mv, rmatvec=rmv
    )

    def test_dimension_mismatch(self):
        linop = LinearOperator([[self.A]])
        with pytest.raises(ValueError) as exc_info:
            linop.matvec(np.ones(linop.shape[1] + 1))
        print(exc_info.value)
        assert str(exc_info.value) == "Dimension mismatch."

    linop = LinearOperator([[A, A]])
    x = np.random.randn(linop.shape[1])
    assert np.allclose(linop.matvec(x), A @ x[:10] + A @ x[10:])
    assert np.allclose(linop.T.rmatvec(x), A @ x[:10] + A @ x[10:])

    linop = LinearOperator([[P, A.T], [A, None]])
    x = np.random.randn(P.shape[1] + A.shape[0])
    assert np.allclose(linop.matvec(x), sp.sparse.bmat([[P, A.T], [A, None]]) @ x)

    linop = LinearOperator([[P, A.T], [A, F]])
    x = np.random.randn(P.shape[1] + A.shape[0])
    assert np.allclose(
        linop.matvec(x),
        sp.sparse.bmat([[P, A.T], [A, 0.1 * sp.sparse.eye(A.shape[0])]]) @ x,
    )


class TestRMatvec:
    I = sp.sparse.eye(10)
    A = sp.sparse.rand(5, 10)
    P = sp.sparse.rand(10, 10)
    P = P.T @ P

    def mv(v):
        return 0.1 * v

    def rmv(v):
        return 0.1 * v

    F = sp.sparse.linalg.LinearOperator(
        (A.shape[0], A.shape[0]), matvec=mv, rmatvec=rmv
    )

    def test_dimension_mismatch(self):
        linop = LinearOperator([[self.A]])
        with pytest.raises(ValueError) as exc_info:
            linop.rmatvec(np.ones(linop.shape[0] + 1))
        print(exc_info.value)
        assert str(exc_info.value) == "Dimension mismatch."

    linop = LinearOperator([[A, A]])
    x = np.random.randn(linop.shape[0])
    assert np.allclose(linop.rmatvec(x), np.concatenate([A.T @ x, A.T @ x]))

    linop = LinearOperator([[P, A.T], [A, None]])
    x = np.random.randn(P.shape[1] + A.shape[0])
    assert np.allclose(linop.rmatvec(x), x @ sp.sparse.bmat([[P, A.T], [A, None]]))

    linop = LinearOperator([[P, A.T], [A, F]])
    x = np.random.randn(P.shape[1] + A.shape[0])
    assert np.allclose(
        linop.rmatvec(x),
        x @ sp.sparse.bmat([[P, A.T], [A, 0.1 * sp.sparse.eye(A.shape[0])]]),
    )


class TestBlockDiag:
    I = sp.sparse.eye(10)
    A = sp.sparse.rand(5, 10)
    P = sp.sparse.rand(10, 10)
    P = P.T @ P

    def mv(v):
        return 0.1 * v

    def rmv(v):
        return 0.1 * v

    F = sp.sparse.linalg.LinearOperator(
        (A.shape[0], A.shape[0]), matvec=mv, rmatvec=rmv
    )

    linop = qss.linearoperator.block_diag([I, I, I])
    x = np.random.randn(linop.shape[1])
    assert np.allclose(linop.matvec(x), x)

    linop = qss.linearoperator.block_diag([F, F, 0.1 * I])
    x = np.random.randn(linop.shape[1])
    assert np.allclose(linop.matvec(x), mv(x))


class TestHstack:
    I = sp.sparse.eye(10)
    A = sp.sparse.rand(5, 10)
    P = sp.sparse.rand(10, 10)
    P = P.T @ P

    def mv(v):
        return 0.1 * v

    def rmv(v):
        return 0.1 * v

    F = sp.sparse.linalg.LinearOperator(
        (A.shape[0], A.shape[0]), matvec=mv, rmatvec=rmv
    )

    linop1 = qss.linearoperator.hstack([F, F])
    x = np.random.randn(linop1.shape[1])
    assert np.allclose(
        linop1.matvec(x), F.matvec(x[: F.shape[0]]) + F.matvec(x[F.shape[0] :])
    )

    linop2 = qss.linearoperator.hstack([linop1, A])
    x = np.random.randn(linop2.shape[1])
    assert np.allclose(
        linop2.matvec(x), linop1.matvec(x[: linop1.shape[1]]) + A @ x[linop1.shape[1] :]
    )


class TestVstack:
    I = sp.sparse.eye(10)
    A = sp.sparse.rand(5, 10)
    P = sp.sparse.rand(10, 10)
    P = P.T @ P

    def mv(v):
        return 0.1 * v

    def rmv(v):
        return 0.1 * v

    F = sp.sparse.linalg.LinearOperator(
        (A.shape[0], A.shape[0]), matvec=mv, rmatvec=rmv
    )

    linop1 = qss.linearoperator.vstack([F, F])
    x = np.random.randn(linop1.shape[1])
    assert np.allclose(
        linop1.matvec(x),
        np.concatenate([F.matvec(x), F.matvec(x)]),
    )

    linop2 = qss.linearoperator.vstack([linop1, A.T])
    x = np.random.randn(linop2.shape[1])
    assert np.allclose(
        linop2.matvec(x),
        np.concatenate([linop1.matvec(x), A.T @ x]),
    )
