import numpy as np
import cvxpy as cp
import scipy as sp
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

    solver = qss.QSS(content, eps_abs=9e-3, eps_rel=1e-4)

    print("Testing a small signal decomposition problem")
    print("  cvxpy:", prob.solve())
    print("  qss:", solver.solve())


def test_nonneg_ls():
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
    print("  qss:", solver.solve())


def test_l1_trend_filtering():
    dim = 1000
    lmda = 1
    y = np.random.rand(dim)

    data = {}
    data["P"] = np.diag(np.concatenate([np.ones(dim), np.zeros(dim - 2)]))
    data["q"] = -np.concatenate([y, np.zeros(dim - 2)])
    data["r"] = 0.5 * y.T @ y
    data["b"] = np.zeros(dim - 2)
    data["g"] = np.concatenate([np.zeros(dim), np.ones(dim - 2)])

    one_zero = np.zeros(dim - 2)
    one_zero[0] = 1
    D = sp.linalg.toeplitz(one_zero, np.concatenate([[1, -2, 1], np.zeros(dim - 3)]))

    data["A"] = np.hstack([D, np.identity(dim - 2)])
    data["P"] = sp.sparse.csc_matrix(data["P"])
    data["A"] = sp.sparse.csc_matrix(data["A"])

    x = cp.Variable(dim)

    objective = cp.Minimize(0.5 * cp.sum_squares(y - x) + lmda * cp.norm(D @ x, 1))
    constraints = []
    prob = cp.Problem(objective, constraints)

    solver = qss.QSS(data, rho=10)

    print("Testing l1 trend filtering")
    print("  cvxpy:", prob.solve())
    print("  qss:", solver.solve())


test_sd_small()
test_nonneg_ls()
test_l1_trend_filtering()
