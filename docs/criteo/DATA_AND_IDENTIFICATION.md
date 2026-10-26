# Data, rights and identification boundary

This extension uses the Criteo Uplift v2.1 release. The [publisher's 2021 paper](https://arxiv.org/html/2111.10106), section 3, describes negative sampling on labels and equalizing advertiser treatment ratios. Section 5.1 checks treatment predictability from features. The [official C2ST implementation](https://raw.githubusercontent.com/criteo-research/large-scale-ITE-UM-benchmark/main/um_util.py) uses logistic classification and simulated Bernoulli labels. These checks do not identify an undisclosed outcome-selection law. This last statement is the project's identification assessment, demonstrated by the exact counterexample below.

## Source manifest

| Item | Value |
|---|---|
| Official mirror | [criteo/criteo-uplift on Hugging Face](https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/criteo-research-uplift-v2.1.csv.gz) |
| Publisher information and terms | [Criteo AI Lab](https://ailab.criteo.com/criteo-uplift-prediction-dataset/) |
| Compressed bytes | 311,422,618 |
| SHA256 | `2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc` |
| Fully validated rows | 13,979,592 |
| Analytical hash cohort | 873,520 rows; 6.2485% of release |
| Outcome / treatment columns | visit, conversion / treatment |
| Allowed predictors | exactly f0 through f11 |

The publisher's current data terms specify **CC BY-NC-SA 4.0**. This project uses the data for noncommercial research, attributes Criteo and the authors, and does not redistribute raw records, sampled records, models or individual predictions. Dataset rights are separate from software rights. The paper's datasheet contains different license wording; this repository does not infer broader rights from it and follows the publisher's more restrictive current terms. For reuse, consult the linked terms. The v2 paper, verified file and checksum define this analysis; the source page also contains older-version descriptions.

Attribution: Eustache Diemert, Artem Betlei, Christophe Renaudin, Massih-Reza Amini, Théophane Gregoir and Thibaud Rahier, *A Large Scale Benchmark for Individual Treatment Effect Prediction and Uplift Modeling* (2021). The publisher also requests attribution to Diemert et al., *A Large Scale Benchmark for Uplift Modeling*, AdKDD/TargetAd (2018).

## Three distinct quantities

1. **Factual prediction:** P(Y=1 | X,T,S=1), measured against held-out released labels.
2. **Release-standardized contrast:** E over released X of differences between released conditional responses, optionally weighted by a frozen allocation. Fitted outcome and assignment models support its estimation; it is a descriptive functional of the selected release.
3. **Conditional reconstruction:** differences between original conditional responses under an explicit retention model, then standardized to the released test X distribution. This is not the original population's average effect.

Here S=1 denotes inclusion in the public release. The observed treatment proportion is not a substitute for a known experimental propensity in a selected sample. Cross-fitting reduces nuisance overfitting; it does not repair selection identification.

## A balanced counterexample

Let X be constant, T randomized, original responses p0=.04 and p1=.06. Keep positives with probability 1 and negatives with probability 1/2 in control, 1/5 in treatment. Then released responses are .076923 and .241935: their difference is .165012, while the original effect is .02. X is independent of T before and after selection. Thus even perfect measured covariate balance cannot establish original-population response effects. This is an exact constructed example, not a claim about the undisclosed Criteo probabilities.

## Scope of the sensitivity family

With r_t=P(S=1|Y=1,T=t,X)/P(S=1|Y=0,T=t,X), assumed constant over X within arm, released odds equal original odds times r_t. Therefore:

```text
p_t(X) = s_t(X) / (r_t*(1-s_t(X)) + s_t(X))
```

The grid [1,2,5,10]² is a declared stress test. It is not a confidence region. Consistency, original conditional exchangeability, overlap, a correct response model and this selection law would be needed for a causal conditional contrast. Unobserved advertiser mixtures and selection on two labels may violate this simplified family. Visit and conversion grids are separate marginal scenarios, not one joint generative mechanism. Selection also changes the X distribution, so this exercise does not reconstruct original-population welfare.

## Audit and leakage protections

The [complete aggregate audit](../../reports/criteo/data_audit.json) records finite-value checks, label counts, numerical ranges and categorical cardinalities. There are 1,626,259 repeated feature-fingerprint rows; fingerprint groups are kept intact across all partitions and nuisance folds. The largest group has 13 rows. The fingerprint is not a verified person ID. Hash collisions were checked against exact feature vectors within the analytical cohort; no collision was observed there. Full-source hash collisions were not exhaustively checked.

Exposure occurs after treatment and is excluded. The other outcome is also excluded. The feature extractor ignores all columns except the explicit twelve-feature whitelist. No outcome-dependent row exclusion or subsampling occurs.
