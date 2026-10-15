# Study protocol — version 1

Frozen before model fitting or inspecting split-specific treatment outcomes, 15 October 2026. This is a versioned analysis protocol, not an external preregistration or an independent replication.

## Question and estimand

Does a learned customer-contact rule improve incremental two-week spending, and scenario contribution after contact costs, over a simple uniform-contact rule? The experiment randomly assigns no email (0), men's email (1) and women's email (2), with nominal propensity 1/3 for each. The target population is the historical experiment population. Each anonymized row is one customer; no identities or longitudinal records are available.

The primary decision uses assumed margin 0.40, cost 0.05 currency units per contacted customer and capacity 1.00. The primary statistical contrast is the locked candidate versus the development-selected best simple policy at that scenario. Report a two-sided 95% confidence interval on the paired difference, without a guaranteed positive result. Costs/margins are assumptions, not measured profitability.

## Split and information boundaries

Permute row positions with NumPy PCG64 seed 20261015. Assign 60% training, 20% development, 20% final evaluation. The split does not depend on treatment or outcome. Preserve original row order, input SHA256 and split hashes. All policy training uses training rows. Development selects policy; final evaluation never changes that choice. No refit after selection in this release.

Allowed predictors: recency, history, mens, womens, zip_code, newbie, channel. Exclude history_segment as a redundant discretization, and exclude segment (assigned treatment), visit, conversion and spend from predictors. The men's/women's labels describe email content and purchasing history, not an observed gender attribute.

## Fixed candidate set

1. No contact; uniform men's email; uniform women's email.
2. A transparent four-segment rule based on prior mens/womens purchasing flags. Estimate arm means in training only, shrunk toward the training arm mean with 200 pseudo-observations.
3. One cross-fitted doubly robust learner: three-fold training-only nuisance fits for arm-conditional mean spend; construct augmented inverse-propensity scores and learn the two contrasts against control. Use fixed histogram gradient-boosting parameters, without hyperparameter search. Fit evaluation nuisance means on training only.

Nuisance models: 100 iterations, learning rate 0.05, seven leaves, minimum leaf 200, L2 10, no early stopping. Effect models: 100 iterations, learning rate 0.03, four leaves, minimum leaf 500, L2 20, no early stopping. Keep negative effect predictions; nonnegative clipping applies only to predicted spending levels. The feature encoder is fitted on training data only; fold-specific encoders are fitted inside nuisance folds.

Choose the best simple policy and best learned candidate on development by the lower 95% confidence bound of incremental scenario value against no contact. Select a learned candidate for the primary comparison regardless of whether it beats the simple policy; the operational recommendation defaults to the simple policy unless final evidence supports the candidate. This is a single prespecified contrast; exploratory curves must not be used for another selection round.

## Evaluation

For action a, the augmented score is Gamma_a = mu_a(X) + I(A=a) * (Y-mu_a(X))/p_a. For a policy pi, value is the mean of Gamma_pi. Evaluate differences with paired customer-level scores. With known randomization and predictions learned outside the evaluation sample, the estimator does not require a correct outcome regression. It still requires consistency, no relevant interference and correct experiment provenance.

Report normal-approximation intervals, a seeded 2,000-resample customer bootstrap for the primary contrast and an inverse-propensity-only sensitivity estimate. Do not assert significance from an exploratory cost/capacity grid. Customer bootstrap intervals condition on the fixed trained policies; they do not include uncertainty from retraining on a new experiment.

Secondary analysis: intention-to-treat arm contrasts; arm/covariate balance; cost grid [0,0.05,0.10,0.25,0.50,1.00], margin [0.20,0.40,0.60], capacity [0.10,0.25,0.50,1.00]. A capacity cap ranks positive predicted incremental scenario value using covariates only, with deterministic row-position tie breaking. Capacitated evaluations are descriptive, conditional on this batch's covariates; displayed intervals are pointwise and do not constitute uniformly valid bands for adaptively chosen scenarios. Uniform policies under a cap use a fixed random priority independent of outcomes.

## Failure checks and limits

Validate data schema, missingness, binary labels and treatment levels; assert split disjointness; test the estimator on analytically known examples and simulated randomized null/heterogeneous treatment settings; test capacity and nonpositive-value decisions. Inspect raw-spend heavy tails and report IPW versus augmented estimates. Do not trim or winsorize the primary outcome after seeing results. Keep all prespecified policies visible, including failures. These data cannot validate transport to present-day India, individual treatment-effect accuracy or realized company savings.

## Release evidence

Require a source manifest, environment lock, clean reproduction command, tests, saved aggregate metrics, split/selection records, technical report, decision memo and a working scenario demonstration. Raw individual data and row-level predictions are excluded from publication. A failed superiority test is a valid empirical finding; invented uplift is not.
