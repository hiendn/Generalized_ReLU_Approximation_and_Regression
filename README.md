# Whole-space approximation and regression with generalized ReLU networks

Code and numerical results for Section 6 of the manuscript by Hien Duy Nguyen,
Jacob Westerhout, TrungTin Nguyen, and Xin Guo.

The experiment compares the adaptive normalised ReLU estimator with two
estimators that use the design density. All observations are simulated from the
model in the manuscript. No external dataset is required.

## Requirements

Use Python 3.11 or later with NumPy and Matplotlib. The dependency versions in
`requirements.txt` were checked with Python 3.12.14 on macOS with Apple silicon.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

On Windows, use `.venv\Scripts\python.exe` in place of `.venv/bin/python`.
Run the commands below from the repository root, without Python's `-O` option.

## Reproduce and verify

Check the stored results and recompute the summary statistics:

```sh
.venv/bin/python verify_results.py
```

This checks all 2,000 replication records and recomputes the 100 rows of mean
risks, Monte Carlo standard errors, and median selected scales. It also checks
the tuning values, feasible block counts, recorded normaliser errors, and run
summary.

Run the implementation diagnostics in a separate directory:

```sh
.venv/bin/python adaptive_regression.py --check --output-dir checks
```

These checks compare the finite normaliser with its hinge representation, kernel
sums with direct calculations, and the validation selector with pairwise loss
comparisons. They also check 4,004 block-count configurations and run small
diagnostic replications. They do not rerun the complete experiment.

Recreate the figure from the stored summaries:

```sh
.venv/bin/python adaptive_regression.py --plot
```

The figure appears in `results/`. Its dotted reference line is
`0.001 * (n/1000)^(-2/3)`. The line's vertical placement is illustrative.
Error bars extend 1.96 Monte Carlo standard errors on each side of the mean
across the 20 replications, computed on the original risk scale.

Rerun the complete experiment in a separate directory:

```sh
.venv/bin/python adaptive_regression.py --run --plot --output-dir rerun
.venv/bin/python verify_results.py --data-dir rerun
```

This preserves the supplied results. The experiment uses 100 logarithmically
spaced integer sample sizes from 100 to 100,000, 20 replications per size, and
10,000 independent test points per replication. The master seed is 20261021.
Each replication's NumPy seed sequence uses the master seed, sample size, and
replication index.

The model is `X ~ N(0,1)`, `f0(x) = exp(-x^2)`, and
`Y = f0(X) * (1 + 0.25 * epsilon)`, with independent `epsilon ~ N(0,1)`.
Each replication splits the observations into training and validation samples.
The normalised estimator evaluates the finite piecewise-affine ReLU normaliser
during validation and prediction. Only the two oracle estimators use the design
density. The median-of-means block count is capped at the largest odd integer
not exceeding the validation sample size.

The stored-result check recomputes summaries from the supplied replication
risks; it does not regenerate their test predictions. Floating-point results
can differ between platforms, particularly where long-double precision differs.

## Source files

- `adaptive_regression.py` implements the estimators, finite normaliser,
  validation selector, simulation, diagnostics, and figure.
- `verify_results.py` checks the stored experiment or a separate full rerun.
- `requirements.txt` records the dependency versions used for verification.

## Reference outputs

The `results/` directory contains the replication-level risks, numerical
summaries, experiment configuration, run summary, and figure in PDF and PNG
formats. It also contains the implementation and stored-result check reports.

At `n = 100000`, the supplied mean risks and Monte Carlo standard errors are:

| Estimator | Mean squared risk | Monte Carlo standard error |
| --- | ---: | ---: |
| Normalised ReLU estimator | 1.23e-05 | 1.35e-06 |
| Density-threshold oracle | 8.55e-05 | 7.18e-06 |
| Unthresholded density oracle | 8.55e-05 | 7.18e-06 |

## Contact

Hien Duy Nguyen: [h.nguyen5@latrobe.edu.au](mailto:h.nguyen5@latrobe.edu.au).
