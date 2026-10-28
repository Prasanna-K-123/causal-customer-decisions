# Reproduce the Criteo extension

The original Hillstrom commands and outputs remain separate. Python 3.12 and the pinned extension environment are required. No account, API key or paid service is needed.

## Obtain the source

Read the [data terms and identification boundary](DATA_AND_IDENTIFICATION.md), then download the official file to a private working directory:

```bash
curl -L --fail 'https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/criteo-research-uplift-v2.1.csv.gz' -o /absolute/private/criteo-uplift-v2.1.csv.gz
```

The pipeline verifies both compressed size and SHA256 before parsing. It stops on a source mismatch. It does not automatically accept a new upstream revision.

## Run from the repository root

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-criteo.txt
export PYTHONPATH=src PYTHONUTF8=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export MPLCONFIGDIR=/absolute/private/matplotlib
export TMPDIR=/absolute/private/tmp
python -m unittest discover -s tests -v
python -m criteo_release.data --source /absolute/private/criteo-uplift-v2.1.csv.gz --scratch /absolute/private/criteo-run
python -m criteo_release.experiment --scratch /absolute/private/criteo-run
python -m criteo_release.verify_outputs --scratch /absolute/private/criteo-run
python -m criteo_release.reporting
```

Create the private temporary directories before running. On Windows use `.venv\Scripts\python.exe`, set the equivalent PowerShell `$env:` variables, and set both TEMP and TMP to the writable private temporary directory. All private paths should be outside the repository. Do not commit raw data, cohort arrays, serialized models or individual predictions. Deserializing the locally generated model pickle is supported only for your own trusted output; never load untrusted pickle files.

The ingestion run covers every source row, then stores the fixed feature-group hash cohort. The experiment runs the public tests and known-truth synthetic experiment, trains the declared candidates, writes the development selection lock, and finally evaluates test outcomes. The reporter reads saved aggregates only.

## A second complete model rerun

Copy the verified cohort and audit to a separate private run directory, preserving their filenames. Run the same experiment with `--scratch /absolute/private/criteo-repeat --report /absolute/private/criteo-repeat/reports`. Then:

```bash
python -m criteo_release.verify_outputs --scratch /absolute/private/criteo-run --repeat /absolute/private/criteo-repeat
```

This compares every scientific numeric field in the real and synthetic results, plus model artifact hashes. Selection-lock timestamps and runtime measurements naturally differ. The original published verification records two complete Windows fixed-model runs using the same verified cohort; source ingestion was run once for those runs. The separately retained [Linux replay](PORTABLE_REPLAY_20261026.md) reacquired and audited the source and refit the fixed models. Its scientific values agree numerically; its model and compressed-cohort bytes are not identical to the Windows artifacts.

## What is checked

- All 54 current tests: 22 Hillstrom tests, 22 extension tests, three process-memory instrumentation tests and seven release-identity checks. The Linux full-replay receipt correctly records 47 tests; the original Windows receipt correctly records the 44 tests available at that time.
- Frozen protocol/config checksums and compressed source hash/size.
- Complete schema, finite values, binary labels, feature-group overlap and cohort collision checks.
- Separate analytical oracles for augmentation, clustered uncertainty and selection inversion.
- Explicit whitelist tests for exposure, treatment and both labels.
- Independent aggregate recomputation from private frozen predictions.
- Exact repeated predictions, deterministic small refits and a complete second model run.

The first preanalysis extension-only test attempt failed because the operating-system temporary directory was unwritable. Setting TEMP/TMP to the designated working directory fixed the environment issue. No model results had been generated; no scientific parameter changed.

## Runtime interpretation

See the original Windows [audit timing](../../reports/criteo/audit_runtime.json), [model timing](../../reports/criteo/model_runtime.json) and [figure timing](../../reports/criteo/reporting_runtime.json). Each contains wall seconds, process CPU seconds and sampled process RSS. The 50 ms monitor can miss shorter peaks; it excludes sibling processes. That recorded machine had eight logical CPUs, with model threads capped at two. These are historical local shared-machine measurements, not hardware-normalized speed comparisons or production latency guarantees. The Linux replay records actual wall/CPU timings but null RSS; it does not reverify the original memory bound.

The 1/16 cohort was frozen to keep fitting below one million rows and observed process memory below a gigabyte while other local work ran. The result should be described as a 14M-row audit with an 874k-row analysis, never as full-14M model training. Exact repeat results on this environment do not guarantee bitwise portability across Python/library versions, operating systems or hardware.

## Information boundaries after release

The test set is now public aggregate evidence and cannot support new confirmatory tuning. Any new encoding, larger cohort, learner, grid, selection rule or uncertainty procedure needs a new versioned protocol and a fresh evaluation design. Fixes to analysis defects require a documented amendment and complete rerun. The v1.0 files and negative results must remain available.

## Restricted execution environments

If the operating system does not expose process RSS, the pipeline continues scientific computation and wall/CPU timing. Its runtime receipt records null RSS, an explicit unavailable/partial status and the measurement error. Missing measurements are never reported as zero memory or as meeting the historical memory bound. This affects instrumentation only; the frozen scientific protocol and configuration are unchanged.

## Publishing a reviewed release

The release workflow runs only on explicit manual dispatch from `main`, with a full reviewed commit SHA and an existing version tag already pointing to that same commit. It checks dispatch, checkout, main and tag identity before and after verification. The verification job has read-only contents access; the separate publication job alone has write access. It never intentionally creates or moves tags, overwrites existing release notes, uploads datasets/models or publishes on an ordinary push.

Release verification runs the public tests and deterministic aggregate rendering. Full raw-source fitting remains a separate, dated review gate; this workflow does not claim a new full-data run. An external tag writer can still race a ref check: workflow concurrency is not platform-enforced tag immutability. Keep tag-writing access controlled and inspect the final release target. A changed main, tag or existing release requires a new review rather than bypassing the guard.

## Explorer rendering and dated runtime evidence

`PYTHONPATH=src python -m criteo_release.reporting --page-only` regenerates `docs/criteo/index.html` from `demo.template.html`, the aggregate result and its recorded test/runtime/verification receipts. It does not rerun models, figures or measurement stages and leaves historical evidence unchanged. The full reporter also regenerates the page after its figures. The explorer separates the original Windows study from the dated Linux replay; test counts and exact-model claims belong to their respective run receipts.
