# QSS: Quadratic-Separable Solver
QSS solves problems of the form 

``` 
minimize    f(x) + g(x)
subject to  Ax = b
```

Where f is a convex quadratic function given by `f(x) = 0.5 x^T P x + q^T x + r` and g is a function that is separable in each entry of x, i.e. `g(x) = g_1(x_1) + ... + g_n(x_n)`.

To use QSS, the user must specify `P`, `q`, `r`, `A`, `b`, as well as the `g_i` from a built-in collection of separable functions. 

## Installation
Clone the repository and run
```
pip3 install ./
```

## Usage

## Development
To create a virtual environment, run
```
python3 -m venv env
```
Activate it with 
```
source env/bin/activate
```
And install `qss` in development mode:
```
pip3 install -e ./ -r requirements.txt
```
Finally, test to make sure the installation worked:
```
python3 tests/test.py
```