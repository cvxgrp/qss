import numpy as np
import scipy as sp
import qdldl
import time
import copy
from qss import precondition
from qss import matrix
from qss import linearoperator
from qss import proximal
from qss import admm
from qss import descent
from qss import util


class QSS:
    def __init__(
        self,
        data,
    ):

        # Checking quadratic part
        if "P" not in data:
            raise ValueError("P matrix must be specified.")

        data_dim = data["P"].shape[0]

        if "q" not in data:
            raise ValueError("q vector must be specified.")
        if "r" not in data:
            raise ValueError("r scalar must be specified.")
        if data["P"].shape[0] != data["P"].shape[1]:
            raise ValueError("P must be a square matrix")
        if len(data["q"]) != data_dim:
            raise ValueError("q dimensions must correspond to P.")

        # Checking constraints
        if "A" in data and data["A"] is not None:
            if "b" not in data or data["b"] is None:
                raise ValueError("Constraint vector not specified.")
            if data["A"].shape[1] != data_dim:
                raise ValueError(
                    "Constraint matrix column number must correspond to P."
                )
            if data["A"].shape[0] != len(data["b"]):
                raise ValueError("A and b dimensions must correspond.")

        if "b" in data and data["b"] is not None:
            if "A" not in data or data["A"] is None:
                raise ValueError("Constraint matrix not specified.")

        # Checking g functions
        ranges = []
        for g in data["g"]:
            if g["g"] not in proximal.G_FUNC_NAMES:
                raise ValueError("Invalid g function name:", g["g"])
            if "args" in g:
                if "weight" in "args":
                    if g["args"]["weight"] < 0:
                        raise ValueError("Weight must be >= 0.")
            if "range" not in g:
                raise ValueError("g function range must be specified.")
            if g["range"][0] < 0:
                raise ValueError("Range out of bounds.")
            if g["range"][1] > data_dim:
                raise ValueError("Range out of bounds.")
            if g["range"][0] > g["range"][1]:
                raise ValueError("Start index must be <= end index.")
            ranges.append(g["range"])
        ranges = sorted(ranges, key=lambda x: x[0])
        for i in range(len(ranges) - 1):
            if ranges[i][1] > ranges[i + 1][0]:
                raise ValueError("g function ranges must not overlap.")

        # Making copies of the input data and storing
        self._data = util.copy_problem_data(data)
        
        if self._data["g"]._is_convex:
            self._relaxed_data = None
        else:
            self._relaxed_data = util.copy_problem_data(data, relax=True)

        # Unscaled data
        self._unscaled_data = {}

        # Scaling information
        self._scaling = {}
        self._reset_scaling()

        # Iterate information
        self._iterates = {}
        self._reset_iterates(random=False)

        # Rho controller
        self._rho_controller = None

        # KKT system information
        self._kkt_system = None

        # User-specified options
        self._options = {}
        self._options["eps_abs"] = None
        self._options["eps_rel"] = None
        self._options["alpha"] = None
        self._options["rho"] = None
        self._options["max_iter"] = None
        self._options["precond"] = None
        self._options["warm_start"] = None
        self._options["reg"] = None
        self._options["use_iter_refinement"] = None
        self._options["descent_method"] = None
        self._options["line_search"] = None
        self._options["algorithms"] = None
        self._options["rho_update"] = None
        self._options["schedule_alpha"] = None
        self._options["random_init"] = None
        self._options["init_seed"] = None
        self._options["verbose"] = None
        self._options["debug"] = None
        return

    def _reset_scaling(self):
        self._scaling["equil_scaling"] = np.ones(self._data["dim"])
        self._scaling["obj_scale"] = 1

    def _reset_iterates(self, random=False):
        if random:
            np.random.seed(self._options["init_seed"])
            self._iterates["x"] = 1000 * np.random.randn(self._data["dim"])
            # self._iterates["y"] = 1000 * np.random.randn(self._data["dim"])
            self._iterates["y"] = np.zeros(self._data["dim"])
        else:
            self._iterates["x"] = np.zeros(self._data["dim"])
            self._iterates["y"] = np.zeros(self._data["dim"])
        self._iterates["obj_val"] = None

    def _reset_kkt_system(self):
        # TODO: should we be checking to see if these keys exist and if so
        # calling `del` on them?
        self._kkt_system.reset()
        self._kkt_system = None

    def solve(
        self,
        eps_abs=1e-5,
        eps_rel=1e-5,
        alpha=1.4,
        rho=0.1,
        max_iter=10000,
        precond=False,
        warm_start=False,
        reg=True,
        use_iter_refinement=False,
        descent_method="momentum",
        line_search=True,
        algorithms=["admm"],
        rho_update="adaptive",
        schedule_alpha=False,
        random_init=False,
        init_seed=1234,
        verbose=False,
        debug=False
    ):

        self._options["eps_abs"] = eps_abs
        self._options["eps_rel"] = eps_rel
        self._options["alpha"] = alpha
        self._options["rho"] = rho
        self._options["max_iter"] = max_iter
        self._options["precond"] = precond
        self._options["warm_start"] = warm_start
        self._options["reg"] = reg
        self._options["use_iter_refinement"] = use_iter_refinement
        self._options["descent_method"] = descent_method
        self._options["line_search"] = line_search
        self._options["algorithms"] = algorithms
        self._options["rho_update"] = rho_update
        self._options["schedule_alpha"] = schedule_alpha
        self._options["random_init"] = random_init
        self._options["init_seed"] = init_seed
        self._options["verbose"] = verbose
        self._options["debug"] = debug

        np.random.seed(1234)

        if self._options["verbose"]:
            util.print_info()
            start_time = time.time()

        # Reset problem parameters if not warm starting
        if not self._options["warm_start"]:
            if self._options["debug"]: print('### DEBUG ###\nresetting iterates\n#############')
            self._reset_iterates(self._options["random_init"])
            self._rho_controller = None

        # Preconditioning
        if self._options["precond"] and not self._data["abstract_constr"]:
            if self._options["verbose"]:
                precond_start_time = time.time()
            self._unscaled_data = copy.deepcopy(self._data)
            # We are now solving for xtilde, where x = equil_scaling * xtilde
            # Note: the below will modify the contents of self._data
            precondition.ruiz(self._data, self._scaling)
            self._iterates["x"] /= self._scaling["equil_scaling"]
            self._iterates["y"] *= self._scaling["obj_scale"]
            if self._options["verbose"]:
                print(
                    "{} {}{}".format(
                        "preconditioning time:".ljust(util.BULLET_WIDTH),
                        format(time.time() - precond_start_time, ".2e"),
                        "s",
                    )
                )

        # Initializing rho controller
        self._rho_controller = util.RhoController(self._data["g"], self._options["rho"])

        # Constructing KKT matrix
        if self._options["verbose"]:
            factorization_start_time = time.time()
        if self._data["abstract_constr"]:
            self._kkt_system = matrix.AbstractKKT(
                self._data["P"], self._data["A"], self._rho_controller
            )
        else:
            self._kkt_system = matrix.KKT(
                self._data["P"], self._data["A"], self._rho_controller
            )
        if self._options["verbose"]:
            print(
                "{} {}{}".format(
                    "initial factorization time:".ljust(util.BULLET_WIDTH),
                    format(time.time() - factorization_start_time, ".2e"),
                    "s",
                )
            )
        max_iter_list = np.atleast_1d(self._options["max_iter"])
        try:
            max_iter_list_int = [int(mxi) for mxi in max_iter_list]
        except ValueError:
            raise ValueError("max_iter should be an integer or a list of integers.")
        # robust int check that works with various numpy int types
        if not np.allclose(max_iter_list_int, max_iter_list):
            raise ValueError("max_iter should be an integer or a list of integers.")
        else:
            max_iter_list = max_iter_list_int
        if self._data["g"]._is_convex and not self._options["rho_update"] == "schedule":
            if self._options["verbose"]:
                print('(standard algorithm)')
            for i, algorithm in enumerate(np.atleast_1d(algorithms)):
                if i == 0 and algorithm == "proj_sd":
                    self._iterates["x"] = sp.sparse.linalg.lsqr(
                        self._data["A"], self._data["b"], atol=1e-12, btol=1e-12
                    )[0]
                self._options["max_iter"] = max_iter_list[i]
                if algorithm == "proj_sd":
                    self._iterates = descent.proj_sd(
                        self._data,
                        self._kkt_system,
                        **self._iterates,
                        **self._scaling,
                        **self._options,
                    )
                elif algorithm == "admm":
                    self._iterates = admm.admm(
                        self._data,
                        self._kkt_system,
                        self._options,
                        self._rho_controller,
                        **self._iterates,
                        **self._scaling,
                    )
                else:
                    raise ValueError("Invalid algorithm specified")

            self._options["max_iter"] = max_iter_list
        elif not self._data["g"]._is_convex:
            if self._options["verbose"]:
                print('(nonconvex warm-start algorithm)')
            # I am only implementing this for ADMM --BM 4/4/23
            if len(max_iter_list) == 1:
                max_iter_list = np.r_[max_iter_list, max_iter_list]
            data_list = [self._relaxed_data, self._data]
            warm_start_list = [False, True]

            for i, data in enumerate(data_list):
                self._options["max_iter"] = max_iter_list[i]
                self._options["warm_start"] = warm_start_list[i]
                self._iterates = admm.admm(
                    data,
                    self._kkt_system,
                    self._options,
                    self._rho_controller,
                    **self._iterates,
                    **self._scaling,
                )

            self._options["max_iter"] = max_iter_list
        else:
            if self._options["verbose"]:
                print('(scheduled algorithm)')
            orig_max_iter = self._options["max_iter"]
            orig_rho = self._options["rho"]
            orig_warm_start = self._options["warm_start"]

            num_rhos = 5
            max_rho = 1e3
            
            self._options["max_iter"] = orig_max_iter // num_rhos
            rho_list = np.logspace(np.log10(orig_rho), np.log10(max_rho), num_rhos)

            for rho in rho_list:
                # TODO: Random start
                self._options["rho"] = rho
                rho_controller = util.RhoController(
                    self._data["g"], self._options["rho"]
                )
                self._kkt_system.update_rho(rho_controller.get_rho_vec())
                self._iterates = admm.admm(
                    self._data,
                    self._kkt_system,
                    self._options,
                    rho_controller,
                    **self._iterates,
                    **self._scaling,
                )

            self._options["max_iter"] = orig_max_iter
            self._options["rho"] = orig_rho
            self._options["warm_start"] = orig_warm_start

        # Clean up preconditioning
        if self._options["precond"] and not self._data["abstract_constr"]:
            self._iterates["x"] *= self._scaling["equil_scaling"]
            self._iterates["y"] /= self._scaling["obj_scale"]
            # TODO: should we del self._data first?
            self._data = self._unscaled_data
            self._reset_scaling()

        # Clean up
        self._reset_kkt_system()

        if self._options["verbose"]:
            util.print_summary(self._iterates["obj_val"], time.time() - start_time)

        return (
            self._iterates["obj_val"],
            np.copy(self._iterates["x"]),
            # np.copy(self  ._iterates["y"] / self._rho_controller.get_rho_vec())
        )
