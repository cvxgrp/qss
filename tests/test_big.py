import numpy as np
import cvxpy as cp
import scipy as sp
import qss
import time
import pytest
from tests import testutil


def test_l1_trend_filtering_big(_verbose):
    lmda = 3000

    # set length of signal
    T = int(1e5)
    # set random seed
    np.random.seed(42)
    # construct signal out of 5 segments
    pwl = np.r_[
        np.linspace(0, 3, num=T // 4),
        np.linspace(3, 1.5, num=T // 6),
        np.linspace(1.5, -3, num=T // 6),
        np.linspace(-3, -2.5, num=T // 4),
        np.linspace(-2.5, 0, num=T // 6 + 2),
    ]
    # add Gaussian noise
    y = pwl + 0.2 * np.random.randn(T)

    m1 = sp.sparse.eye(m=T - 2, n=T, k=0)
    m2 = sp.sparse.eye(m=T - 2, n=T, k=1)
    m3 = sp.sparse.eye(m=T - 2, n=T, k=2)
    D = m1 - 2 * m2 + m3

    # CVXPY
    x = cp.Variable(T)
    objective = cp.Minimize(0.5 * cp.sum_squares(y - x) + lmda * cp.norm(D @ x, 1))
    constraints = []
    prob = cp.Problem(objective, constraints)
    t0 = time.time()

    # QSS
    data = {}
    data["P"] = sp.sparse.diags(np.concatenate([np.ones(T), np.zeros(T - 2)]))
    data["q"] = -np.concatenate([y, np.zeros(T - 2)])
    data["r"] = 0.5 * y.T @ y
    data["b"] = np.zeros(T - 2)
    data["g"] = [{"g": "abs", "range": (T, 2 * T - 2)}]
    data["A"] = sp.sparse.hstack([lmda * D, -sp.sparse.identity(T - 2)])
    solver = qss.QSS(data, alpha=1.8, rho=0.005, verbose=_verbose)

    qss_res = testutil.compare_qss_cvxpy(prob, solver)

    print("Real objective values:")
    print(
        0.5 * np.linalg.norm(y - x.value) ** 2
        + lmda * np.linalg.norm(D @ x.value, ord=1)
    )
    print(
        0.5 * np.linalg.norm(y - qss_res[:T]) ** 2
        + lmda * np.linalg.norm(D @ qss_res[:T], ord=1)
    )


def test_lp_big(_verbose):
    np.random.seed(1234)
    dim = 100
    constr_dim = 30

    c = 10 * (np.random.rand(dim) - 0.3)
    A = sp.sparse.random(constr_dim, dim, density=0.1, format="csc")
    b = 2 * np.random.rand(constr_dim)

    data = {}
    data["P"] = sp.sparse.csc_matrix((dim + constr_dim, dim + constr_dim))
    data["q"] = np.concatenate([c, np.zeros(constr_dim)])
    data["r"] = 0
    data["A"] = sp.sparse.hstack([A, -sp.sparse.eye(constr_dim)])
    data["b"] = b
    data["g"] = [
        {"g": "is_pos", "range": (0, dim)},
        {"g": "is_pos", "args": {"scale": -1}, "range": (dim, dim + constr_dim)},
    ]

    # CVXPY
    x = cp.Variable(dim)
    objective = cp.Minimize(c @ x)
    constraints = [A @ x <= b, x >= 0]
    prob = cp.Problem(objective, constraints)

    # QSS
    solver = qss.QSS(data, verbose=_verbose)

    qss_res = testutil.compare_qss_cvxpy(prob, solver)


def test_quadratic_control_big(_verbose):
    np.random.seed(1234)
    n = 50
    m = 20
    T = 30

    Q = sp.sparse.random(n, n, density=0.1, format="csc")
    R = sp.sparse.random(m, m, density=0.1, format="csc")
    A = sp.sparse.random(n, n, density=0.3, format="csc")
    B = sp.sparse.random(n, m, density=0.3, format="csc")
    xinit = 10 * (np.random.rand(n) - 0.5)

    Q = Q @ Q.T
    R = R @ R.T

    # CVXPY
    obj = 0
    constraints = []
    x = cp.Variable((n, T + 1))
    u = cp.Variable((m, T + 1))

    for t in range(T + 1):
        obj += 0.5 * (cp.quad_form(x[:, t], Q) + cp.quad_form(u[:, t], R))
        if t == 0:
            constraints.append(x[:, t] == xinit)
        else:
            constraints.append(x[:, t] == A @ x[:, t - 1] + B @ u[:, t - 1])

    constraints += [cp.norm(u, "inf") <= 1]
    objective = cp.Minimize(obj)
    prob = cp.Problem(objective, constraints)
    print(prob.solve())

    # QSS
    data = {}
    data["P"] = sp.sparse.block_diag(
        [Q for i in range(T + 1)] + [R for i in range(T + 1)]
    )
    data["q"] = np.zeros(n * (T + 1) + m * (T + 1))
    data["r"] = 0
    Abig = sp.sparse.block_diag([A for i in range(T)])
    Abig = sp.sparse.hstack([Abig, sp.sparse.csc_matrix((n * T, n))])
    Bbig = sp.sparse.block_diag([B for i in range(T)])
    Bbig = sp.sparse.hstack([Bbig, sp.sparse.csc_matrix((n * T, m))])
    Ibig = -sp.sparse.eye(n * T, n * T)
    Ibig = sp.sparse.hstack([sp.sparse.csc_matrix((n * T, n)), Ibig])
    constr_mat = sp.sparse.hstack([Abig + Ibig, Bbig])
    constr_mat = sp.sparse.vstack(
        [
            sp.sparse.hstack(
                [sp.sparse.eye(n), sp.sparse.csc_matrix((n, n * T + m * (T + 1)))]
            ),
            constr_mat,
        ]
    )
    data["A"] = constr_mat
    data["b"] = np.concatenate([xinit, np.zeros(n * T)])
    data["g"] = [
        {
            "g": "is_bound",
            "args": {"weight": 1, "scale": 0.5, "shift": -0.5},
            "range": (n * (T + 1), n * (T + 1) + m * (T + 1)),
        }
    ]

    solver = qss.QSS(data, verbose=_verbose)

    assert prob.solve() == pytest.approx(solver.solve()[0], rel=1e-2)
