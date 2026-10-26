# Causal Customer Decisions

[![Public research checks](https://github.com/Prasanna-K-123/causal-customer-decisions/actions/workflows/verify.yml/badge.svg)](https://github.com/Prasanna-K-123/causal-customer-decisions/actions/workflows/verify.yml)

**Does personalized targeting earn its complexity?** A three-arm randomized experiment, cross-fitted causal learning and an untouched evaluation set connect statistical evidence to a business decision.

**Finding:** the learned policy did not establish superiority over a simple campaign. Under the predefined 40% margin and $0.05 contact-cost scenario, its paired difference was **−$0.049 per customer**, 95% CI **[−$0.193, +$0.095]**. The recommendation is to retain the development-selected uniform men's email policy for this historical population and scenario. This is not a claim of realized profit or present-day campaign performance.

| Evidence | Result |
|---|---|
| Original randomized experiment | 64,000 customers; three arms |
| Training / development / evaluation | 38,400 / 12,800 / 12,800 |
| Evaluation design | Policies frozen before final outcomes; known treatment probabilities |
| Uncertainty | Paired confidence interval; 2,000-resample bootstrap; IPW sensitivity |
| Integrity | Source checksums, split hashes, mathematical tests and a versioned protocol |

[Technical report](reports/TECHNICAL_REPORT.md) · [Decision memo](reports/DECISION_MEMO.md) · [Interactive scenario explorer](https://prasanna-k-123.github.io/causal-customer-decisions/) · [Study protocol](docs/PROTOCOL.md) · [Data provenance](docs/DATA.md)

![Held-out policy values](reports/policy_comparison.svg)

## What is implemented

- A checked download and data-audit pipeline; raw data stays outside the repository.
- A transparent historical-purchase segment policy and a three-fold doubly robust learner with fixed model complexity.
- Paired policy-value evaluation, known-propensity sensitivity and explicit cost/capacity assumptions.
- An offline scenario explorer using aggregate results, with uncertainty and exploratory-analysis limits visible.

The contribution is the complete evidence chain from randomized observations to a defensible operational recommendation. The underlying estimators and benchmark dataset are established; no new statistical algorithm or commercial impact is claimed.

## Automated public checks

The automated checks run on public files without private data, credentials or model refitting. All 22 tests cover the original research checks plus input-contract, numerical-boundary and row-order invariance checks. Raw-source model reproduction is separate.

The [21 October validation revision](docs/INPUT_CONTRACT_REVISION_20261021.md) rejects malformed treatment/action labels before integer conversion and requires finite scenario inputs. Independent complete reruns of the original and revised code reproduce all five published aggregate outputs byte for byte in the pinned environment.

## Reproduce

Python 3.12. Install the pinned environment, run tests, run the analysis and generate the reports:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
PYTHONPATH=src python -m unittest discover -s tests -v
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=src python -m customer_policy.experiment
PYTHONPATH=src python -m customer_policy.reporting
```

Open `docs/index.html` in a browser. The explorer works offline; its report links open GitHub. The first analysis run obtains the checksum-verified public dataset; no account or API key is required. If the upstream source is unavailable, the loader stops rather than silently substituting data. Floating-point results may vary slightly across platforms.

## Limits and sources

The data comes from Kevin Hillstrom's MineThatData experiment. Campaign labels refer to merchandise, not observed customer gender. Margin, costs and capacity are scenarios. More than 99% of evaluation customers have zero spending, so policy differences are imprecise. No raw-data redistribution license has been established; obtain data separately under the provider's terms. Documentation cites the data creator and methodological sources.

Project: Prasanna K · Economics student at Madras School of Economics.

## 26 October 2026 verification

A fresh download and complete fixed-model rerun on Windows reproduce the scientific values in all five checked aggregate outputs within absolute/relative tolerance `1e-10`. Platform text bytes and the environment field differ; no byte-identical cross-platform claim is made. See [the dated verification guide](docs/VERIFICATION_20261026.md). The reported original experiment remains the reference study.
