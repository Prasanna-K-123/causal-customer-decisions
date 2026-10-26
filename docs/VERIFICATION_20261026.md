# Current verification and portable setup — 26 October 2026

Status: working historical research prototype. A completed artifact does not certify its presenter's personal mastery.

A fresh acquisition of the documented Hillstrom HTTPS mirror returned SHA-256 `bab6578f60db5d792f1c2372c502f029152a5249cf5ea84390f3b7f885d7234f`. The loader also verified the pinned MD5, schema, 64,000 rows and input constraints. No source rows are redistributed.

The complete fixed experiment ran in Python 3.12.14 on Windows with the pinned requirements, `OMP_NUM_THREADS=2` and `OPENBLAS_NUM_THREADS=1`. All 22 public tests passed. The selected comparison remains DR learner versus uniform men's email; the selected recommendation remains the uniform policy. The difference is −0.04864597809142988 per customer under the declared economic scenario, with normal interval [−0.19274814088257053, 0.09545618469971079]. This is conditional benchmark value, not realized business profit.

The current run compared `results.json`, `selection.json`, `scenario_grid.csv`, `test_arm_summary.csv` and `training_balance.csv` to the original public release. Scientific values matched with absolute/relative tolerance 1e-10; only the `environment` object was excluded from the JSON comparison. Text line endings and environment metadata prevent a cross-platform byte-identity claim. No models, costs, split or selection criterion were changed.

## Windows PowerShell

Use a fresh checkout; `.gitattributes` now preserves exact stored source bytes even when Git's global line-ending conversion is enabled. Do not normalize frozen research files by hand.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:PYTHONPATH = 'src'
$env:OMP_NUM_THREADS = '2'
$env:OPENBLAS_NUM_THREADS = '1'
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m customer_policy.experiment
.\.venv\Scripts\python.exe -m customer_policy.reporting
```

Use a permitted writable temporary directory if the environment's default temp location is restricted. No global security or system changes are required. Preserve the original published evidence when reproducing in a separate checkout. Compare numerical meaning explicitly; do not claim newly generated files are byte-identical if only their numbers match.

## Interpretation that remains important

The final sample contains 110 spenders out of 12,800 customers, so uncertainty is material. The primary full-capacity comparison uses fixed trained policies. Exploratory capacity-constrained policies also depend on the evaluation batch's covariates; their fixed-score intervals do not include retraining or reallocating a new batch. A separate randomized deployment and present-day transport assessment would be needed before organizational use. Existing dated reports retain their historical wording; current status is a student research portfolio, not an earned postgraduate qualification or client deployment.
