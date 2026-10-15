# Decision memo: customer contact policy

**Recommendation:** retain the development-selected uniform men's email campaign for the defined historical population and primary cost scenario. Personalized targeting has not established incremental superiority.

The study used a 64,000-customer randomized experiment and a 12,800-customer untouched evaluation set. It compared uniform campaigns, a transparent purchase-segment policy and a cross-fitted doubly robust learner. Policy choices were fixed using separate development data.

The learned candidate's estimated difference from the simple comparator was −$0.049 per customer; its 95% interval was [−$0.193, $0.095]. A paired 2,000-resample bootstrap and an IPW-only estimate reached the same qualitative conclusion. The candidate contacted 94.9% of customers, but the reduced contact count did not establish superior scenario value.

These figures assume a 40% margin and $0.05 contact cost. Actual margin and contact costs were not provided. The observation is incremental spending; financial contribution is a scenario calculation. Most customers spent zero, so uncertainty is material.

The practical decision is to avoid paying for a more complex targeting system on the strength of an unproven uplift claim. A modern deployment would need a fresh experiment, actual operating costs and evidence that treatment effects transport to the current population. The accompanying scenario explorer illustrates how assumptions change the analysis; it does not create new confirmatory evidence.

Evidence: [technical report](TECHNICAL_REPORT.md), [machine-readable results](results.json), [frozen selection](selection.json), [data audit](data_audit.json).
