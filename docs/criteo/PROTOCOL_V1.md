# Criteo scale and selection sensitivity — protocol v1.0

Frozen 26 October 2026 before parsing any outcome rows or fitting models for this extension. This is a local, time-stamped analysis protocol, not an externally registered preregistration. The separate Hillstrom protocol, code and scientific outputs remain the reference randomized case and are unchanged.

## Questions and claim boundary

1. On a large, label-independent sample of the released Criteo data, how do fixed simple and flexible learners perform at factual response prediction and ranking a **release-standardized response contrast**?
2. What computation is required on this local machine?
3. Under explicit, unidentified outcome-retention assumptions, how do reconstructed conditional response contrasts and hypothetical allocation decisions change?

The third question is sensitivity analysis, not recovery of the publisher's selection mechanism. Criteo's published negative sampling on labels and advertiser rebalancing mean that 0.85 is an empirical released treatment proportion, not an established known randomization probability for these released records. Covariate balance does not establish potential-outcome exchangeability after outcome-dependent selection. No Criteo campaign ROI, unqualified causal policy value, original-population treatment effect, individual effect accuracy or deployment claim is permitted.

## Sources and input contract

Use only version 2.1, compressed size 311422618 bytes, SHA256 `2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc`. Expected 13979592 data rows. Official publisher mirror: https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/criteo-research-uplift-v2.1.csv.gz . Source description: https://ailab.criteo.com/criteo-uplift-prediction-dataset/ . Mechanism: Diemert et al. (2021), https://arxiv.org/html/2111.10106 , sections 3 and 5.1. Source failures or checksum mismatches stop execution; do not silently substitute versions. Raw data, sampled rows, trained models and individual predictions stay outside public files.

Exact columns: f0,...,f11,treatment,conversion,visit,exposure. Predictors are exactly f0,...,f11. Treatment is a conditioning variable, never a ranking predictor. Exposure is post-treatment and excluded. Neither outcome can predict the other. Parse feature numbers in float64 with round-trip conversion; normalize signed zero before fingerprinting; convert labels to integer only after validating finite binary values.

Stream the complete file in chunks of 200000. Audit row count, column order, missing/non-finite values, binary label counts, feature minima/maxima/means/standard deviations, exact categorical cardinalities (f1,f3,f4,f5,f6,f8,f9,f11), full feature-fingerprint multiplicities and maximum group size. Continuous cardinality is not claimed. Check conversion-without-visit and control exposure as diagnostics; do not remove rows based on their values. Any malformed schema/nonfinite/binary error aborts.

## Cohort and partitions, independent of labels

Compute a 64-bit fingerprint of the ordered twelve normalized float64 features with pandas.util.hash_pandas_object(index=False) in pinned pandas 2.2.3. This is a feature-vector group, **not a verified person ID**. A collision conservatively groups distinct feature vectors together. Check for such collisions in the analytical cohort using exact vector equality.

Mix fingerprint bits with SplitMix64 and distinct fixed salts:
- cohort salt 2026102601; include mix % 16 == 0 (about 874000 rows expected);
- partition salt 2026102602; mix % 10000 < 6000 training, 6000–7999 development, otherwise final test;
- cross-fit salt 2026102603; mix % 3 gives the training nuisance fold.

All identical normalized feature vectors must stay together through cohort selection, train/dev/test and nuisance folds. Retain multiplicities; do not pick different samples after observing sparsity. Store sorted group hashes, row-order cohort content hash and split hashes. The 1/16 fraction bounds simultaneous model fitting below one million rows while retaining an expected ~175000-row final evaluation; full-source validation still covers all 14M rows. Report actual achieved sizes and positive-event counts. This is a group-hash sample of the release, not a fresh experiment.

## Fixed models and finite budget

Use only training rows. No parameter search, early stopping, class weighting, feature screening or post-test refit.

- Arm-specific constants, for factual-response reference only.
- Pooled logistic regression: StandardScaler fitted on training X, treatment added as a separate input, C=1, lbfgs, max_iter=400, tolerance 1e-7; no treatment interactions.
- T learner: two HistGradientBoostingClassifier models, one per arm, 80 iterations, learning_rate=.05, max_leaf_nodes=15, min_samples_leaf=200, l2_regularization=10, max_bins=255, early_stopping=False, random_state=20261026.
- DR ranking learner: three groupwise cross-fits of the same T outcome models and a StandardScaler/logistic X-to-treatment model (same logistic settings). Construct release augmented response-contrast targets with fitted e(X), clipped to [.05,.95]. Fit HistGradientBoostingRegressor on the targets, 80 iterations, learning_rate=.03, max_leaf_nodes=7, min_samples_leaf=500, l2_regularization=20, max_bins=255, early_stopping=False, random_state=20261026.
- Full-training T models and fitted logistic e(X) supply the common independent evaluation nuisance estimates. Their evaluation residuals must not train any ranking.

Histograms treat anonymized numerical projections as ordered numeric inputs; this is a transparent baseline choice, not semantic recovery of categorical features. No assertion of state-of-the-art performance.

## Development selection and final-touch policy

Primary outcome visit; secondary conversion, with its own fixed models and visibly secondary results. For each outcome select one ranking among pooled logistic, T learner and DR learner by the development integrated excess release response over the fixed budget grid [0,.05,.1,.2,.4,.6,.8,1]. The score is the trapezoid integral of mean((a_b-b)*psi); higher is better. Ties within 1e-12 prefer pooled logistic, then T, then DR.

Here psi = m1-m0 + T*(Y-m1)/e - (1-T)*(Y-m0)/(1-e), using training-only nuisances. This estimates a functional of the **released** conditional response distribution, with fitted/clipped propensity and the usual nuisance-consistency conditions; it is not identified original campaign uplift. Also report the unaugmented inverse-weighted analogue as a model-dependence diagnostic.

Rank decreasing score with fingerprint as deterministic tie-break. Allocate exactly the requested fraction in expectation; if a fingerprint group straddles the budget, assign its members the same fractional allocation. The resulting policy is a stochastic capacity policy at that boundary, with no outcome used to break ties. No-response allocation is the b=0 baseline; random allocation at the same budget is the b benchmark.

Write a selection lock with protocol/source/split hashes, selected models, development metrics and a hash of every fitted training model **before final test outcomes are evaluated**. Evaluate all fixed candidates on test and flag the preselected one. Test outcomes may inform only the report, no model/scenario/selection revision. Corrections to an implementation defect require a dated amendment and complete rerun; record the reason and retain the previous receipt.

## Evaluation and uncertainty

Primary contrast: development-selected visit ranking minus pooled logistic in integrated excess release response. If pooled logistic wins, the contrast is exactly zero; do not substitute another comparison. Report all candidates, per-budget mean release-standardized contrast versus no allocation and versus same-budget random allocation, paired intervals, and treatment/positive counts within the top-20% weighted allocation.

Factual response: arm-constant, pooled logistic and T models get overall and armwise log loss, Brier score, ROC AUC and average precision. The DR ranking has no separate probability model; do not relabel the common T nuisances as its independent response performance. Report learned-propensity range, clipping count, factual treatment AUC/log loss versus the train-proportion constant. These are descriptive diagnostics, not a test proving randomization.

For paired scalar contrasts and pointwise budget curves use a fingerprint-cluster sandwich standard error of the sample mean: center row contributions, sum within each group, apply G/(G-1), divide by n squared, then take square root. Use normal 95% intervals. Primary paired contrast also receives 1000 seeded Gaussian cluster-multiplier draws (seed 20261026), using the same centered cluster sums. This conditions on trained models and the batch's covariates/rank cutoffs; it excludes training/model-selection uncertainty and is not a simultaneous band. Unknown person/advertiser dependence remains unaddressed. Sparse conversions and treatment imbalance must be visible. Do not call an exploratory secondary contrast confirmatory.

## Outcome-selection sensitivity

For each outcome separately define r_t = P(S=1|Y=1,T=t,X)/P(S=1|Y=0,T=t,X), assumed constant over X within arm. Grid r0,r1 in [1,2,5,10]; these are declared scenarios, not estimated parameters or verified bounds. For released conditional response s_t(X), invert Bayes:
p_t(X) = s_t(X) / [r_t*(1-s_t(X)) + s_t(X)].
This follows by comparing positive/negative odds. At r=1 the release response is unchanged; higher positive relative retention reduces the implied original response. Use fitted full-training T response probabilities on test covariates, with no outcome-dependent scenario choice.

Under consistency, original conditional exchangeability, adequate overlap, correct response models and this selection law, p1-p0 is a conditional treatment contrast. Average only over the **released evaluation covariate distribution**, not the unidentified original population. The publisher's joint two-label/advertiser mechanism need not lie in this reduced family. Separate marginal visit and conversion sensitivity grids do not specify one joint two-outcome selection law.

For all 16 r pairs report mean reconstructed contrast, positive-contrast fraction, top-20% conditional net response gain and selected share under response-unit costs visit [0,.001,.005,.01], conversion [0,.0001,.0005,.001]. Only positive net scores can be allocated and capacity is .2; score is p1-p0-cost. These are model-based scenario calculations, with neither causal nor sampling confidence claims. Original X distribution and selection probabilities are unidentified; no original-population values or monetary gains are shown.

## Known truth and engineering checks

Before the real-data final evaluation, require tests of source/schema failure, exact feature whitelist, exposure/other-label invariance, label-independent grouping, cross-split and cross-fit group disjointness, row-order invariance, capacity/tie/negative-score rules, clustered paired standard-error oracle, exact augmented-score double-robust identities and odds inversion at boundary/invalid values.

Use an exact finite-stratum selection oracle with known response surfaces/selection ratios to show naive released contrasts differ while inversion recovers the declared truth; include a constant-X example where covariate balance holds despite selected response distortion.

Also generate independent synthetic training 60000 and evaluation 40000 observations, seed 20261026, four independent uniform[-1,1] features padded with eight zeros. e=.5; p0=expit(-3+.7*x0-.4*x1); tau=.035*tanh(2*x0)-.015*x2; p1=clip(p0+tau,.001,.95). Train the same fixed learners; report PEHE and true fixed-budget gain plus augmented-score estimates using the known e. This is a known-truth implementation check, not evidence of Criteo effect identification. Repeat a deterministic small fit and verify identical predictions; independently recompute aggregate results from private frozen predictions.

## Runtime, environment, artifacts

Python 3.12; original pinned scientific packages plus extension-only psutil 7.2.2 for sampled process memory. Cap OMP, OpenBLAS, MKL, NumExpr threads at 2. Record CPU/platform/library versions, wall time and process CPU time per stage; sample process RSS every .05 seconds, recording observed stage/whole-process peaks. Memory measurements are sampled, not exact allocator peaks. Include checksum/audit, training per model, prediction/evaluation, synthetic and reporting runtimes. Local timings on a shared machine are not throughput guarantees, hardware-normalized comparisons or production service benchmarks. No paid API/service is used.

Public: protocol/hash, code/tests, audit aggregates, selection lock, metrics/uncertainty, runtime/environment, plots generated from saved metrics, honest limitations and source attribution. Private: cohort/cache, individual predictions/models, logs and defense walkthrough. Preserve all original Hillstrom scientific files and findings.
