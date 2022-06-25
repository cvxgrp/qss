from sqlite3 import adapt
import numpy as np
import scipy as sp
import qdldl
import time
import copy
from qss import precondition
from qss import matrix
from qss import proximal
from qss import admm
from qss import descent
from qss import util


class QSS(object):
    def __init__(
        self,
        data,
    ):

        # Checking quadratic part
        if "P" not in data:
            raise ValueError("P matrix must be specified.")

        self._data = {}
        self._data["dim"] = data["P"].shape[0]

        if "q" not in data:
            raise ValueError("q vector must be specified.")
        if "r" not in data:
            raise ValueError("r scalar must be specified.")
        if data["P"].shape[0] != data["P"].shape[1]:
            raise ValueError("P must be a square matrix")
        if len(data["q"]) != self._data["dim"]:
            raise ValueError("q dimensions must correspond to P.")

        # Checking constraints
        if "A" in data and data["A"] is not None:
            if "b" not in data or data["b"] is None:
                raise ValueError("Constraint vector not specified.")
            if data["A"].shape[1] != self._data["dim"]:
                raise ValueError(
                    "Constraint matrix column number must correspond to P."
                )
            if data["A"].shape[0] != len(data["b"]):
                raise ValueError("A and b dimensions must correspond.")

        if "b" in data and data["b"] is not None:
            if "A" not in data or data["A"] is None:
                raise ValueError("Constraint matrix not specified.")

        # Checking g functions
        for g in data["g"]:
            if g["g"] not in proximal.g_funcs:
                raise ValueError("Invalid g function name:", g["g"])
            if "args" in g:
                if "weight" in "args":
                    if g["args"]["weight"] < 0:
                        raise ValueError("Weight must be >= 0.")
            if "range" not in g:
                raise ValueError("g function range must be specified.")
            if g["range"][0] < 0:
                raise ValueError("Range out of bounds.")
            if g["range"][1] > self._data["dim"]:
                raise ValueError("Range out of bounds.")
            if g["range"][0] > g["range"][1]:
                raise ValueError("Start index must be <= end index.")

        # Making copies of the input data and storing
        self._data["P"] = data["P"].copy()
        self._data["q"] = np.copy(data["q"])
        self._data["r"] = data["r"]
        self._data["g"] = copy.deepcopy(data["g"])

        if ("A" not in data) or (data["A"] is None) or (data["A"].nnz == 0):
            # TODO: get rid of this placeholder when QSS is more object-oriented, i.e.,
            # when all problem data is passed around together.
            # I'm using the placeholder for now to avoid littering precondition.py with
            # 'if' statements.
            self._data["A"] = sp.sparse.csc_matrix((1, self._data["dim"]))
            self._data["b"] = np.ones(1)
            self._data["has_constr"] = False
            self._constr_dim = 0
        else:
            self._data["A"] = data["A"].copy()
            self._data["b"] = np.copy(data["b"])
            self._data["has_constr"] = True
            self._data["constr_dim"] = data["A"].shape[0]

        # Scaling information
        self._scaling = {}
        self._scaling["equil_scaling"] = np.ones(self._data["dim"])
        self._scaling["obj_scale"] = 1

        # Iterate information
        self._iterates = {}
        self._iterates["x"] = np.zeros(self._data["dim"])
        self._iterates["y"] = np.zeros(self._data["dim"])
        self._iterates["obj_val"] = None

        # KKT system information
        self._kkt_info = {}
        self._kkt_info["quad_kkt"] = None
        self._kkt_info["quad_kkt_unreg"] = None
        self._kkt_info["F"] = None

        # User-specified options
        self._options = {}
        self._options["eps_abs"] = None
        self._options["eps_rel"] = None
        self._options["alpha"] = None
        self._options["rho"] = None
        self._options["adaptive_rho"] = None
        self._options["max_iter"] = None
        self._options["precond"] = None
        self._options["reg"] = None
        self._options["use_iter_refinement"] = None
        self._options["polish"] = None
        self._options["sd_init"] = None
        self._options["proj_sd"] = None
        self._options["verbose"] = None
        return

    def solve(
        self,
        eps_abs=1e-4,
        eps_rel=1e-4,
        alpha=1.4,
        rho=0.1,
        adaptive_rho=True,
        max_iter=np.inf,
        precond=True,
        reg=True,
        use_iter_refinement=True,
        polish=False,
        sd_init=False,
        proj_sd=False,
        verbose=False,
    ):

        self._options["eps_abs"] = eps_abs
        self._options["eps_rel"] = eps_rel
        self._options["alpha"] = alpha
        self._options["rho"] = rho
        self._options["adaptive_rho"] = adaptive_rho
        self._options["max_iter"] = max_iter
        self._options["precond"] = precond
        self._options["reg"] = reg
        self._options["use_iter_refinement"] = use_iter_refinement
        self._options["polish"] = polish
        self._options["sd_init"] = sd_init
        self._options["proj_sd"] = proj_sd
        self._options["verbose"] = verbose

        if self._options["verbose"]:
            util.print_info()
            start_time = time.time()

        # Preconditioning
        if self._options["precond"]:
            if self._options["verbose"]:
                precond_start_time = time.time()
            # We are now solving for xtilde, where x = equil_scaling * xtilde
            # Note: the below will modify the contents of self._data
            precondition.ruiz(self._data, self._scaling)
            if self._options["verbose"]:
                print(
                    "### Preconditioning finished in {} seconds. ###".format(
                        time.time() - precond_start_time
                    )
                )

        # Using steepest descent to initialize ADMM (only for unconstrained problems)
        """
        if self._sd_init and not has_constr:
            if self._verbose:
                init_start_time = time.time()
            xk, sd_iter = polish.steepest_descent(
                g, xk, P, q, r, equil_scaling, obj_scale, ord=2, max_iter=10
            )
            zk = xk
            uk = -(P @ xk + q) / rho
            if self._verbose:
                print(
                    "### Starting point found in {} seconds using {} iterations. ###".format(
                        time.time() - init_start_time, sd_iter
                    )
                )
        """

        # Constructing KKT matrix
        if self._options["verbose"]:
            factorization_start_time = time.time()
        if self._data["has_constr"]:
            self._kkt_info["quad_kkt_unreg"] = matrix.build_kkt(
                self._data["P"],
                self._data["A"],
                0,
                self._options["rho"],
                self._data["dim"],
                self._data["constr_dim"],
            )
            self._kkt_info["quad_kkt"] = matrix.build_kkt(
                self._data["P"],
                self._data["A"],
                -1e-7,
                self._options["rho"],
                self._data["dim"],
                self._data["constr_dim"],
            )
        else:
            self._kkt_info["quad_kkt"] = self._data["P"] + self._options[
                "rho"
            ] * sp.sparse.identity(self._data["dim"])

        self._kkt_info["F"] = qdldl.Solver(self._kkt_info["quad_kkt"])

        if self._options["verbose"]:
            print(
                "### Factorization finished in {} seconds. ###".format(
                    time.time() - factorization_start_time
                )
            )

        """
        if self._proj_sd:
            if self._proj_sd == True:
                method = "momentum"
            else:
                method = self._proj_sd
            x = sp.sparse.linalg.lsqr(A, b, atol=1e-12, btol=1e-12)[0]
            x_proj_sd, proj_sd_iter = polish.proj_sd(
                x,
                g,
                P,
                q,
                r,
                A,
                b,
                F,
                equil_scaling,
                obj_scale,
                method=method,
                max_iter=self._max_iter,
            )
            print("Projected SD took {} iterations".format(proj_sd_iter))
            print("Time taken: {}".format(time.time() - solve_start_time))
            return (
                util.evaluate_objective(
                    P, q, r, g, x_proj_sd, obj_scale, equil_scaling
                ),
                equil_scaling * x_proj_sd,
            )
            # Preparing for ADMM:
            # zk = x_proj_sd
            # uk = -(P @ x_proj_sd + q) / rho
        """

        self._iterates = admm.admm(
            self._data,
            self._kkt_info,
            **self._iterates,
            **self._scaling,
            **self._options
        )

        return self._iterates["obj_val"], self._iterates["x"]

        """
        # Polishing (only works with no constraints for now)
        if (not has_constr) and self._polish:
            zk1, polish_iter = polish.steepest_descent(
                g, zk1, P, q, r, equil_scaling, obj_scale
            )
            polish_obj_val = util.evaluate_objective(
                P, q, r, g, zk1, obj_scale, equil_scaling
            )
            if self._verbose:
                util.print_status(
                    "plsh", polish_obj_val, -1, -1, rho, solve_start_time
                )
                print("    iterations:", polish_iter)
        """

        # if self._proj_sd:
        #     # x = sp.sparse.linalg.lsqr(A, b)[0]
        #     x_proj_sd, proj_sd_iter = polish.proj_sd(
        #         xk1, g, P, q, r, A, b, F, equil_scaling, obj_scale
        #     )
        #     print("Projected SD took {} iterations".format(proj_sd_iter))
        #     print("Time taken: {}".format(time.time() - solve_start_time))
        #     return (
        #         util.evaluate_objective(
        #             P, q, r, g, x_proj_sd, obj_scale, equil_scaling
        #         ),
        #         equil_scaling * x_proj_sd,
        # )
