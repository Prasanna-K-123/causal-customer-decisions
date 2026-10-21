# Causal evaluation input contract — 21 October 2026

The evaluation API previously converted treatment and policy-action labels to integers before checking their domain. Fractional labels such as 0.9, 1.9 and 2.9 could consequently be treated as 0, 1 and 2. A NaN contact cost could also yield an all-no-contact allocation instead of rejecting the invalid scenario. The checked research data and the fixed experiment configuration do not contain these inputs.

The revision validates labels before conversion. Treatment/action vectors must be one-dimensional, aligned, real numerical labels equal to 0, 1 or 2. Integral floating-point labels remain admissible. Scenario parameters must be finite real scalars: margin and contact capacity in [0, 1], contact cost nonnegative. Numerical predictions, augmented scores and priorities must be finite and have the declared dimensions.

Tie keys must be aligned, unique real numbers or strings, giving a deterministic capacity cutoff independent of customer row order. Uniform policies validate their arm, integer batch size, capacity and priorities. Summaries require at least two finite one-dimensional scores and confidence strictly between zero and one; bootstrap repetitions must be an integer of at least two. Arithmetic that exceeds finite numerical precision raises ValueError.

## Verification

All 22 tests pass: the ten original research tests and twelve adversarial contract/invariance tests. The added cases exercise fractional/unknown labels, NaN/infinite/complex inputs, shape misalignment, invalid confidence/bootstrap parameters, ambiguous tie keys, overflow, exact equal-value capacity cutoffs and joint row permutation.

Both the unchanged v1.0.2 baseline and this revision independently reran the complete fixed experiment from the same checksum-verified 64,000-row source cache. Each run repeated the original training, development selection, final evaluation and scenario generation. Five outputs match the original publication byte for byte: results.json, selection.json, scenario_grid.csv, test_arm_summary.csv and training_balance.csv.

The configuration, protocol, data loader, model definitions and experiment orchestration retain their original hashes. There is no new candidate, hyperparameter search or final-result reselection. The historical recommendation and its uncertainty remain unchanged.

The [machine-readable receipt](../reports/input_contract_revision_20261021.json) records the output and source hashes. Both runs used the same host and exact pinned Python 3.12.14 environment. This is an implementation/reproduction check, not an external or cross-platform review.
