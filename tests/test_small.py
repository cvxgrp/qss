import numpy as np
import cvxpy as cp
import scipy as sp
import qss
import pytest


def test_nonneg_ls(_verbose):
    np.random.seed(1234)
    p = 100
    n = 500
    G = np.random.rand(n, p)
    h = np.random.rand(n)

    data = {}
    data["P"] = G.T @ G
    data["q"] = -h.T @ G
    data["r"] = 0.5 * h.T @ h
    data["A"] = np.zeros((1, p))
    data["b"] = np.zeros(1)
    data["g"] = [{"g": "is_pos", "args": {}, "range": (0, p)}]

    data["P"] = sp.sparse.csc_matrix(data["P"])
    data["A"] = sp.sparse.csc_matrix(data["A"])

    # CVXPY
    x = cp.Variable(p)
    objective = cp.Minimize(0.5 * cp.sum_squares(G @ x - h))
    constraints = [x >= 0]
    prob = cp.Problem(objective, constraints)

    # QSS
    solver = qss.QSS(data)
    qss_result, x_qss = solver.solve(verbose=_verbose)

    assert prob.solve() == pytest.approx(qss_result, rel=1e-2)


def test_quantile_regression(_verbose):
    np.random.seed(1235)
    p = 100
    n = 500
    G = np.random.rand(n, p)
    h = np.random.rand(n)

    tau = 0.25

    data = {}
    data["P"] = G.T @ G
    data["q"] = -h.T @ G
    data["r"] = 0.5 * h.T @ h
    data["A"] = np.zeros((1, p))
    data["b"] = np.zeros(1)
    data["g"] = [{"g": "quantile", "args": {"tau": tau}, "range": (0, p)}]

    data["P"] = sp.sparse.csc_matrix(data["P"])
    data["A"] = sp.sparse.csc_matrix(data["A"])

    # CVXPY
    x = cp.Variable(p)
    objective = cp.Minimize(
        0.5 * cp.sum_squares(G @ x - h) + 0.5 * cp.norm(x, 1) + cp.sum((tau - 0.5) * x)
    )
    constraints = []
    prob = cp.Problem(objective, constraints)

    # QSS
    solver = qss.QSS(data)
    qss_result, x_qss = solver.solve(verbose=_verbose)

    assert prob.solve() == pytest.approx(qss_result, rel=1e-2)


def test_huber_regression(_verbose):
    np.random.seed(1235)
    p = 100
    n = 500
    G = np.random.rand(n, p)
    h = np.random.rand(n)

    M = 1

    data = {}
    data["P"] = G.T @ G
    data["q"] = -h.T @ G
    data["r"] = 0.5 * h.T @ h
    data["A"] = np.zeros((1, p))
    data["b"] = np.zeros(1)
    data["g"] = [{"g": "huber", "args": {"M": M}, "range": (0, p)}]

    data["P"] = sp.sparse.csc_matrix(data["P"])
    data["A"] = sp.sparse.csc_matrix(data["A"])

    # CVXPY
    x = cp.Variable(p)
    objective = cp.Minimize(0.5 * cp.sum_squares(G @ x - h) + cp.sum(cp.huber(x)))
    constraints = []
    prob = cp.Problem(objective, constraints)

    # QSS
    solver = qss.QSS(data)
    qss_result, x_qss = solver.solve(verbose=_verbose)

    assert prob.solve() == pytest.approx(qss_result, rel=1e-2)


def test_l0_ls(_verbose):
    np.random.seed(1234)
    p = 100
    n = 500
    G = np.random.rand(n, p)
    h = np.random.rand(n)

    data = {}
    data["P"] = G.T @ G
    data["q"] = -h.T @ G
    data["r"] = 0.5 * h.T @ h
    data["A"] = np.zeros((1, p))
    data["b"] = np.zeros(1)
    data["g"] = [{"g": "card", "args": {"weight": 0.1}, "range": (0, p)}]

    data["P"] = sp.sparse.csc_matrix(data["P"])
    data["A"] = sp.sparse.csc_matrix(data["A"])

    solver = qss.QSS(data)
    qss_result, x_qss = solver.solve(
        algorithms=["admm"], max_iter=[1000], verbose=_verbose
    )

    # Regression test as CVXPY can't solve this problem
    # assert pytest.approx(qss_result, rel=1e-2) == 19.945084341967263


def test_int_ls(_verbose):
    np.random.seed(1234)
    p = 100
    n = 500
    G = np.random.rand(n, p) - 0.5
    h = -100 * np.random.rand(n)

    data = {}
    data["P"] = G.T @ G
    data["q"] = -h.T @ G
    data["r"] = 0.5 * h.T @ h
    data["A"] = np.zeros((1, p))
    data["b"] = np.zeros(1)
    data["g"] = [{"g": "is_int", "args": {}, "range": (0, p)}]

    data["P"] = sp.sparse.csc_matrix(data["P"])
    data["A"] = sp.sparse.csc_matrix(data["A"])

    solver = qss.QSS(data)
    qss_result, x_qss = solver.solve(
        algorithms=["admm"], max_iter=[1000], rho=1e-5, verbose=_verbose
    )

    # Regression test as CVXPY can't solve this problem
    assert pytest.approx(qss_result, rel=1e-2) == 694195.2486268306


def test_finite_set_regression(_verbose):
    np.random.seed(1234)
    p = 100
    n = 500
    G = np.random.rand(n, p) - 0.5
    h = -100 * np.random.rand(n)

    data = {}
    data["P"] = G.T @ G
    data["q"] = -h.T @ G
    data["r"] = 0.5 * h.T @ h
    data["A"] = np.zeros((1, p))
    data["b"] = np.zeros(1)
    data["g"] = [
        {"g": "is_bool", "args": {}, "range": (0, 10)},
        {"g": "is_finite_set", "args": {"S": {0, 1, 2, 3, 4, 5}}, "range": (10, 100)},
    ]

    data["P"] = sp.sparse.csc_matrix(data["P"])
    data["A"] = sp.sparse.csc_matrix(data["A"])

    solver = qss.QSS(data)
    qss_result, x_qss = solver.solve(
        algorithms=["admm"], max_iter=[1000], rho=1e-10, verbose=_verbose
    )

    # Regression test as CVXPY can't solve this problem
    assert pytest.approx(qss_result, rel=1e-2) == 792202.3536915448


def test_l1_trend_filtering(_verbose):
    np.random.seed(1234)
    dim = 5000
    lmda = 1
    y = np.random.rand(dim)

    data = {}
    data["P"] = sp.sparse.diags(np.concatenate([np.ones(dim), np.zeros(dim - 2)]))
    data["q"] = -np.concatenate([y, np.zeros(dim - 2)])
    data["r"] = 0.5 * y.T @ y
    data["b"] = np.zeros(dim - 2)
    data["g"] = [{"g": "abs", "range": (dim, 2 * dim - 2)}]

    m1 = sp.sparse.eye(m=dim - 2, n=dim, k=0)
    m2 = sp.sparse.eye(m=dim - 2, n=dim, k=1)
    m3 = sp.sparse.eye(m=dim - 2, n=dim, k=2)
    D = m1 - 2 * m2 + m3

    data["A"] = sp.sparse.hstack([D, -sp.sparse.identity(dim - 2)])

    # CVXPY
    x = cp.Variable(dim)
    objective = cp.Minimize(0.5 * cp.sum_squares(y - x) + lmda * cp.norm(D @ x, 1))
    constraints = []
    prob = cp.Problem(objective, constraints)

    # QSS
    solver = qss.QSS(data)
    qss_result, x_qss = solver.solve(verbose=_verbose)

    assert prob.solve() == pytest.approx(qss_result, rel=1e-2)


def test_lp(_verbose):
    np.random.seed(1234)
    dim = 100
    constr_dim = 30

    c = 10 * np.random.rand(dim)
    A = sp.sparse.random(constr_dim, dim, density=0.1, format="csc")
    b = np.random.rand(constr_dim)

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
    solver = qss.QSS(data)
    qss_result, x_qss = solver.solve(verbose=_verbose)

    assert prob.solve() == pytest.approx(qss_result, abs=1e-2)


def test_quadratic_control(_verbose):
    np.random.seed(1234)
    n = 5
    m = 2
    T = 30

    Q = sp.sparse.random(n, n, density=0.1, format="csc")
    R = sp.sparse.random(m, m, density=0.1, format="csc")
    A = sp.sparse.random(n, n, density=0.3, format="csc")
    B = sp.sparse.random(n, m, density=0.3, format="csc")
    xinit = 10 * np.random.rand(n)

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

    solver = qss.QSS(data)
    qss_result, x_qss = solver.solve(verbose=_verbose)

    assert prob.solve() == pytest.approx(qss_result, rel=1e-2)
