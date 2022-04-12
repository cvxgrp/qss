import numpy as np
import cvxpy as cp
import scipy as sp
from matplotlib import pyplot as plt
import time
import pickle
import qss

# Testing signal decomposition problem
def test_sd_small():
    content = pickle.load(open("canonicalized_problem.pkl", "rb"))
    content["P"] = content["P"].tocsc()
    content["A"] = content["A"].tocsc()

    P = content["P"]
    A = content["A"]
    b = content["b"]
    g = content["g"]
    q = np.zeros(P.shape[0])
    r = 0

    content["q"] = q
    content["r"] = r

    dim = P.shape[0]
    x = cp.Variable(dim)

    objective = cp.Minimize(
        0.5 * cp.quad_form(x, P) + q @ x + r + cp.norm(x[np.where(g == 1)], 1)
    )
    constraints = [A @ x == b]
    prob = cp.Problem(objective, constraints)

    solver = qss.QSS(content, eps_abs=1e-4, eps_rel=1e-4, rho=0.6)

    print("Testing a small signal decomposition problem")
    print("  cvxpy:", prob.solve())
    print("  qss:", solver.solve()[0])


def test_nonneg_ls():
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
    data["g"] = 2 * np.ones(p)

    data["P"] = sp.sparse.csc_matrix(data["P"])
    data["A"] = sp.sparse.csc_matrix(data["A"])

    x = cp.Variable(p)

    objective = cp.Minimize(0.5 * cp.sum_squares(G @ x - h))
    constraints = [x >= 0]
    prob = cp.Problem(objective, constraints)

    solver = qss.QSS(data, eps_abs=1e-4, eps_rel=1e-4, rho=2)

    print("Testing nonnegative least squares")
    print("  cvxpy:", prob.solve())
    print("  qss:", solver.solve()[0])


def test_l1_trend_filtering():
    np.random.seed(1234)
    dim = 5000
    lmda = 1
    y = np.random.rand(dim)

    data = {}
    data["P"] = sp.sparse.diags(np.concatenate([np.ones(dim), np.zeros(dim - 2)]))
    data["q"] = -np.concatenate([y, np.zeros(dim - 2)])
    data["r"] = 0.5 * y.T @ y
    data["b"] = np.zeros(dim - 2)
    data["g"] = np.concatenate([np.zeros(dim), np.ones(dim - 2)])

    one_zero = np.zeros(dim - 2)
    one_zero[0] = 1
    D = sp.linalg.toeplitz(one_zero, np.concatenate([[1, -2, 1], np.zeros(dim - 3)]))

    data["A"] = sp.sparse.hstack([D, -sp.sparse.identity(dim - 2)])

    x = cp.Variable(dim)

    objective = cp.Minimize(0.5 * cp.sum_squares(y - x) + lmda * cp.norm(D @ x, 1))
    constraints = []
    prob = cp.Problem(objective, constraints)

    solver = qss.QSS(data, rho=10)

    print("Testing l1 trend filtering")
    print("  cvxpy:", prob.solve())
    print("  qss:", solver.solve()[0])


def test_l1_trend_filtering_big():
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

    x = cp.Variable(T)
    objective = cp.Minimize(0.5 * cp.sum_squares(y - x) + lmda * cp.norm(D @ x, 1))
    constraints = []
    prob = cp.Problem(objective, constraints)
    t0 = time.time()
    print("Testing big l1 trend filtering")
    print("  cvxpy:", prob.solve())
    print("  cvxpy took", time.time() - t0, "seconds")

    data = {}
    data["P"] = sp.sparse.diags(np.concatenate([np.ones(T), np.zeros(T - 2)]))
    data["q"] = -np.concatenate([y, np.zeros(T - 2)])
    data["r"] = 0.5 * y.T @ y
    data["b"] = np.zeros(T - 2)
    data["g"] = np.concatenate([np.zeros(T), np.ones(T - 2)])
    data["A"] = sp.sparse.hstack([lmda * D, -sp.sparse.identity(T - 2)])
    # solver = qss.QSS(data, eps_abs=1e-4, eps_rel=1e-5, rho=0.4)
    solver = qss.QSS(data, eps_abs=1e-4, eps_rel=1e-4, alpha=1.8, rho=0.005)
    t0 = time.time()
    qss_result, x_qss = solver.solve()
    print("  qss:", qss_result)
    print("  qss took", time.time() - t0, "seconds")
    print("Real objective values:")
    print(
        0.5 * np.linalg.norm(y - x.value) ** 2
        + lmda * np.linalg.norm(D @ x.value, ord=1)
    )
    print(
        0.5 * np.linalg.norm(y - x_qss[:T]) ** 2
        + lmda * np.linalg.norm(D @ x_qss[:T], ord=1)
    )


def test_quadratic_control():
    n = 5
    m = 2
    T = 30

    np.random.seed(1234)

    Q = sp.sparse.random(n, n, density=0.1, format="csc")
    R = sp.sparse.random(m, m, density=0.1, format="csc")
    A = sp.sparse.random(n, n, density=0.3, format="csc")
    B = sp.sparse.random(n, m, density=0.3, format="csc")
    xinit = 10 * np.random.rand(n)

    Q = Q @ Q.T
    R = R @ R.T

    # Solving with CVXPY
    obj = 0
    constraints = []
    x = cp.Variable((n, T + 1))
    u = cp.Variable((m, T + 1))

    for t in range(T):
        obj += 0.5 * (cp.quad_form(x[:, t], Q) + cp.quad_form(u[:, t], R))
        if t == 0:
            constraints.append(x[:, t] == xinit)
        else:
            constraints.append(x[:, t] == A @ x[:, t - 1] + B @ u[:, t - 1])

    constraints += [cp.norm(u, "inf") <= 1]
    objective = cp.Minimize(obj)
    prob = cp.Problem(objective, constraints)
    print(prob.solve())

    # Solving with QSS
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
    data["g"] = np.zeros(n * (T + 1) + m * (T + 1))

    print(data["P"].shape)
    print(data["q"].shape)
    print(data["A"].shape)
    print(data["b"].shape)
    solver = qss.QSS(data, eps_abs=1e-4, eps_rel=1e-4, rho=0.4)
    qss_result, x_qss = solver.solve()
    print(qss_result)


test_sd_small()
test_nonneg_ls()
test_l1_trend_filtering()
test_l1_trend_filtering_big()
"""
test_quadratic_control()
"""
