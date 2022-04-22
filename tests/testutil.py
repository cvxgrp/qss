import numpy as np
import cvxpy as cp
import scipy as sp
import time
import qss


def compare_qss_cvxpy(cp_prob, qss_solver):
    print("Comparing CVXPY to QSS.")
    print("Starting CVXPY")
    t0 = time.time()
    cp_res = cp_prob.solve()
    cp_time = time.time() - t0

    print("Starting QSS")
    t0 = time.time()
    qss_res = qss_solver.solve()
    qss_time = time.time() - t0

    print("-----------------------------------")
    print("                       CVXPY       QSS")
    print("-----------------------------------")
    print("time (s)         ", "{:10.3f}".format(cp_time), "{:10.3f}".format(qss_time))
    print("objective value: ", "{:10.3f}".format(cp_res), "{:10.3f}".format(qss_res[0]))

    return qss_res[1]
