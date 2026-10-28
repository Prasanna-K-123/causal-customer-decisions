# Criteo replay — 26 October 2026

This receipt separates the original Windows study from a subsequent Linux reproduction. The historical reports remain unchanged. The Linux run acquired the pinned source, audited all 13,979,592 rows, rebuilt the 873,520-row cohort, fit every fixed candidate, selected on development data and evaluated the held-out split. No protocol, model specification, split rule or selection criterion changed.

## Reproduced evidence

The [replay directory](../../reports/criteo/replay_20261026/) retains aggregate outputs, selection and verification receipts, and actual runtime measurements. It contains no individual observations, predictions or fitted models. The [comparison receipt](../../reports/criteo/replay_20261026/comparison.json) records file hashes and the original-versus-replay comparison.

| Check | Result |
|---|---|
| Source SHA256 and compressed size | Identical: `2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc`; 311,422,618 bytes |
| Audit numerical fields | All 103 identical; split content, row and fingerprint hashes identical |
| Real scientific fields | 2,567 compared; maximum absolute difference `7.2267747341925315e-12` |
| Synthetic scientific fields | 55 compared; maximum absolute difference `3.469446951953614e-18` |
| Scientific comparison threshold | `1e-10` absolute, as used by the original repeated-run verifier |
| Development choices | Same two-model learner for visit and conversion; six selection scores differ by at most `4.616532850443278e-15` |
| Independent aggregate verifier | 190 checks passed; maximum discrepancy `5.551115123125783e-17` |
| Reloaded replay-model predictions | Exact repeat within this run |
| Public tests at replay | 47 passed, including three optional-memory instrumentation checks |

The primary visit comparison still does not establish superiority over pooled logistic. Secondary conversion evidence and all identification/uncertainty limitations retain their original interpretation. Small arithmetic agreement is not evidence that the released sample identifies original-campaign effects.

## What was not identical

Both runs used Python 3.12.14 with NumPy 2.3.5, pandas 2.2.3, SciPy 1.17.0, scikit-learn 1.8.0, matplotlib 3.10.8 and psutil 7.2.2. The original system was Windows/MSVC; the replay was Linux/Clang. The model pickle and compressed cohort have different hashes. Exact split-content hashes establish logical split identity, not container-byte identity. This review did not isolate each serialization difference or compare every original row-level prediction; it does not claim exact cross-platform model equivalence.

The Linux process-memory interface was unavailable. Its receipts contain null RSS, zero successful RSS samples and the observed `NoSuchProcess` error, while wall/CPU timing continued. Null is not zero memory and does not establish a memory bound. Shared-machine timings are observations, not production benchmarks or hardware-normalized comparisons.

## How to inspect or reproduce

Use the [reproduction commands](REPRODUCE.md) in a separate checkout with private source/model paths. Compare the new audit, split identities, locked development decisions and scientific values against both the original and this dated replay; investigate any disagreement rather than changing reference values. The existing independent verifier requires the privately regenerated predictions to recompute the arithmetic. The public aggregate receipt records that completed check; aggregate files alone cannot rerun prediction-level verification.

The `1e-10` threshold predates this replay and is much smaller than the displayed six-decimal result resolution. It is a comparison rule for this fixed study, not a proven error bound for every platform. Future changes to modelling, uncertainty or allocation require a new versioned evaluation design. No new fit is implied by the later documentation and deterministic-rendering repairs.
