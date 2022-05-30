The `tests/` folder is comprised of the following test suites:
- `test_input.py`: Tests on input error catching, e.g., poorly formatted `g`, matrix dimension mismatch, etc. 
- `test_proximal.py`: Tests on all functions related to proximal operator and subdifferential calculations. 
- `test_small.py`: Small solver accuracy tests. QSS output is compared to CVXPY output (or, in the case of nonconvex problems, QSS output is compared to previous saved QSS output).
- `test_big.py`: Larger tests comparing QSS output to CVXPY output. No assertions are made in this test suite as results can sometimes differ significantly. 

Run all tests (with output printed):
```
pytest tests/ -s
```

Run all tests with QSS `verbose` flag set to `True`:
```
pytest tests/ -s --_verbose=True
```