# QSS: Quadratic-Separable Solver
QSS solves problems of the form 

$$\begin{equation*} \begin{array}{ll} \text{minimize} & (1/2) x^T P x + q^T x + r + g(x) \\\\ \text{subject to} & Ax = b \end{array} \end{equation*}$$

where $x \in \bf{R}^n$ is the decision variable being optimized over. The
objective is defined by a positive definite matrix $P \in \bf{S}^n_+$, a vector
$q \in \bf{R}^n$, a scalar $r \in \bf{R}$, and a $g$ that is separable in the
entries of $x$, i.e., $g$ can be written as 
$$g(x) = \sum_{i=1}^n g_i(x_i).$$
The constraints are defined by a matrix $A \in \bf{R}^{m \times n}$ and a vector
$b \in \bf{R}^m$. 

To use QSS, the user must specify $P$, $q$, $r$, $A$, $b$, as well as the $g_i$ from a built-in collection of separable functions. 

## Installation
```
pip install qss
```

## Usage
After installing `qss`, import it with
```python
import qss
```
This will expose the QSS class which is used to instantiate a solver object:
```python
solver = qss.QSS(data)
```
Use the `solve()` method when ready to solve:
```python
results = solver.solve(eps_abs=1e-5,
                       eps_rel=1e-5,
                       alpha=1.4,
                       rho=0.1,
                       max_iter=np.inf,
                       precond=True,
                       warm_start=False,
                       reg=True,
                       use_iter_refinement=False,
                       verbose=False,
                       )
```

### Parameters
- `data`: dictionary with the following keys:
    - `'P'`, `'q'`, `'r'`, `'A'`, `'b'` specify the quadratic part of the objective and the linear constraint as in the problem formulation above. `'P'` and `'A'` should be `scipy.sparse` CSC matrices or QSS `LinearOperator`s (see below), `'q'` and `'b'` should be `numpy` arrays,  and `'r'` should be a scalar. `'A'` and `'b'` can be excluded from `data` or set to `None` if the linear equality constraints are not needed. 
    - `'g'` is a list of separable function definitions. Each separable function is declared as a dictionary with the following keys:
        - `'g'`: string that corresponds to a valid separable function name (see below for a list of supported functions).
        - `'args'`: `'weight'` (default 1), `'scale'` (default 1), `'shift'` (default 0) allow the `'g'` function to be applied in a weighted manner to a shifted and scaled input. Some functions take additional arguments, see below. 
        - `'range'`: tuple specifying the start index and end index that the function should be applied to.
    
        Note that the zero function will be applied to any indices that don't have another function specified for them.
- `eps_abs`: scalar specifying absolute tolerance.
- `eps_abs`: scalar specifying relative tolerance.
- `alpha`: scalar specifying overstep size.
- `rho`: scalar specifying ADMM step size.
- `max_iter`: maximum number of ADMM iterations to perform.
- `precond`: boolean specifying whether to perform matrix equilibration.
- `warm_start`: boolean specifying whether to warm start upon a repeat call of
  `solve()`.
- `reg`: boolean specifying whether to regularize KKT matrix. May fail on certain problem instances if set to `False`.
- `use_iter_refinement`: boolean, only matters if `reg` is `True`. Helps mitigate some of the accuracy loss due to regularization. 
- `verbose`: boolean specifying whether to print verbose output.

### Returns
A list containing the following:
- `objective`: the objective value attained by the solution found by `qss`. 
- `solution`: the solution vector.

### Separable functions
The following convex separable functions are supported ( $\mathcal{I}$ is the $\\{0, \infty\\}$ indicator function):
Function | Parameters | $g_i(x)$
--- | --- | ---
`zero` | | $0$
`abs` | | $\|x\|$
`is_pos` | | $\mathcal I(x \geq 0)$
`is_neg` | | $\mathcal I(x \leq 0)$
`is_bound` | `lb`: lower bound (default 0), `ub`: upper bound (default 1) | $\mathcal I(l \leq x \leq u)$
`is_zero` | | $\mathcal I(x = 0)$
`pos` | | $\max\\{x, 0\\}$
`neg` | | $\max\\{-x, 0\\}$
`quantile` | `tau`: scalar in $(0, 1)$ | $0.5 \|x\| + (\tau - 0.5) x$
`huber` | `M`: positive scalar | $x^2 \text{ if } \|x\| \leq M, 2M\|x\| - M^2 \text{ else}$

The following nonconvex separable functions are supported:
Function | Parameters | $g_i(x)$
--- | --- | ---
`card` | | $0$ for $x = 0$; $1$ for $x \neq 0$
`is_int` | | $\mathcal I(x \text{ is an integer})$
`is_finite_set` | `S`: Python list of scalars | $\mathcal I(x \in S)$
`is_bool` | | $\mathcal I(x \in \\{0,1\\})$

The `t` (weight), `a` (scale), `b` (shift) parameters are used to shift and scale the above as follows: `t * g(ax - b)`.

#### Example
Applying the Huber function to a shifted version of the first 100 entries:
```python
[{"g": "huber", "args": {"M": 2, "shift": -5}, "range": (0, 100)}]
```

### Abstract linear operators
QSS comes with built-in support for abstract linear operators via the `qss.linearoperator.LinearOperator` class (hereafter referred to simply as `LinearOperator`).

The easiest way to build a `LinearOperator` is via its constructor. The argument to the constructor should be a list of lists representing a block matrix, in which each block is one of the following:
- SciPy sparse matrix or
- [`scipy.sparse.linalg.LinearOperator`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LinearOperator.html) or
- `qss.linearoperator.LinearOperator` or
- `None`.

As an example, a constraint matrix `A` could be built as follows:
```python
from qss.linearoperator import LinearOperator

A = LinearOperator([
    [None, F, -I],
    [I, I, None]
])
```
Where `F` is a `scipy.sparse.linalg.LinearOperator` that implements the Fourier transform and `I` is a SciPy sparse identity matrix. 

There are several helper functions available to facilitate the creation of `LinearOperator`s, all accessible through `qss.linearoperator`: 
- `block_diag(D)`: Returns a block diagonal `LinearOperator` from `D`, a list of linear operators (SciPy sparse matrix, `scipy.sparse.linalg.LinearOperator`, or `qss.linearoperator.LinearOperator`).
- `hstack(D)`: Horizontally concatenates list of linear operators `D` into a single `LinearOperator`.
- `vstack(D)`: Vertically concatenates a list of linear operators `D` into a single `LinearOperator`.

Left and right matrix multiplication between a `LinearOperator` and a NumPy array is supported. Multiplication between `LinearOperator`s is currently not supported. Matrix-vector multiplication between a `LinearOperator` `F` and a NumPy array `v` can be achieved with `F.matvec(v)` or `F @ v`. Multiplication with the adjoint of `F` can be achieved with `F.rmatvec(v)` or `v @ F`. 

Note that solve times may be slower when `LinearOperator`s are involved. If either `P` or `A` is a `LinearOperator`, the linear KKT system central to the QSS algorithm is solved indirectly, as opposed to directly via a factorization. 


### Example
Nonnegative least squares is a problem of the form

$$\begin{equation*} \begin{array}{ll} \text{minimize} & (1/2) \Vert Gx - h \Vert_2^2 \\\\ \text{subject to} & x \geq 0. \end{array} \end{equation*} $$

`qss` can be used to solve this problem as follows:
```python
import numpy as np
import scipy as sp
import qss

p = 100
n = 500
G = sp.sparse.random(n, p, density=0.2, format="csc")
h = np.random.rand(n)

data = {}
data["P"] = G.T @ G
data["q"] = -h.T @ G
data["r"] = 0.5 * h.T @ h
data["g"] = [{"g": "is_pos", "range": (0, p)}] # Enforce x >= 0

solver = qss.QSS(data)
objective, x = solver.solve()
print(objective)
```

## Development
To create a virtual environment, run
```
python3 -m venv env
```
Activate it with 
```
source env/bin/activate
```
Clone the `qss` repository, `cd` into it, and install `qss` in development mode:
```
pip install -e ./ -r requirements.txt
```
Finally, test to make sure the installation worked:
```
pytest tests/
```
