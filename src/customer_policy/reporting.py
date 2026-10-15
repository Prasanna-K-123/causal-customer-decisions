"""Generate reports and an offline interactive view from measured aggregate results."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABELS={'none':'No contact','all_mens':"Uniform men's email",'all_womens':"Uniform women's email",'segment':'Purchase-segment policy','dr_learner':'Doubly robust learner'}


def render(root:Path) -> None:
    reports=root/'reports';docs=root/'docs'
    result=json.loads((reports/'results.json').read_text())
    selection=json.loads((reports/'selection.json').read_text())
    primary=result['primary'];diff=primary['paired_difference'];boot=primary['bootstrap']
    names=list(result['policies'])
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none'})
    fig,ax=plt.subplots(figsize=(9,4.2),layout='constrained')
    for i,name in enumerate(names):
        s=result['policies'][name]['incremental_vs_none']
        ax.errorbar(s['estimate'],i,xerr=[[s['estimate']-s['lower']],[s['upper']-s['estimate']]],fmt='o',color='#0b7669' if name==primary['comparator'] else '#50637d',capsize=4,markersize=7)
    ax.axvline(0,color='#b8c1cc',lw=1);ax.set_yticks(range(len(names)),[LABELS[x] for x in names]);ax.invert_yaxis()
    ax.set_xlabel('Estimated incremental scenario value per customer ($)')
    ax.set_title('Held-out policy values versus no contact\n40% margin · $0.05/contact · 12,800 customers · pointwise 95% intervals',loc='left',weight='bold',fontsize=11,pad=16)
    ax.spines[['top','right','left']].set_visible(False);ax.grid(axis='x',alpha=.15)
    fig.savefig(reports/'policy_comparison.svg');fig.savefig(reports/'policy_comparison.png',dpi=160);plt.close(fig)
    rows=[]
    for name in names:
        s=result['policies'][name]['incremental_vs_none'];contact=result['policies'][name]['contact_fraction']
        rows.append(f"| {LABELS[name]} | {s['estimate']:.4f} | [{s['lower']:.4f}, {s['upper']:.4f}] | {contact:.1%} |")
    table='\n'.join(rows)
    conclusion='The learned candidate did not establish superiority over the simple campaign.'
    report=f'''# Causal Customer Decisions — empirical report

Prasanna K · 15 October 2026

{conclusion} The predefined comparison estimates a difference of **${diff['estimate']:.4f} per customer**, with a paired 95% normal interval **[${diff['lower']:.4f}, ${diff['upper']:.4f}]**. The customer bootstrap interval is **[${boot['lower']:.4f}, ${boot['upper']:.4f}]**. Under the specified assumptions, retain the development-selected uniform men's email policy; this does not establish that uniform campaigns are universally optimal.

## Decision and data

The decision is whether individualized email targeting warrants replacing a simple campaign. Kevin Hillstrom's historical randomized experiment contains 64,000 customers assigned to no email, men's email or women's email. Outcome: actual spending in the following two weeks. Men/women refer to merchandise and campaign content; no customer gender variable is observed.

The input was checked against the documented HTTPS loader and checksum. Every row is retained. A row-position permutation, independent of treatment and outcomes, assigns 38,400 customers to training, 12,800 to development and 12,800 to final evaluation. Published artifacts contain aggregates, not customer-level data. See [data provenance](../docs/DATA.md) and [the analysis protocol](../docs/PROTOCOL.md).

## Method and selection

Three training-only cross-fitting folds estimate arm-specific mean spending with fixed histogram gradient boosting. Known randomization probabilities produce augmented scores; two regularized regressions learn treatment contrasts against no contact. A separate baseline uses four historical-purchase segments with shrinkage toward training arm means. No hyperparameter search or post-test refitting was performed.

The development-selected simple comparator is **{LABELS[primary['comparator']]}**. The selected learned candidate is **{LABELS[primary['candidate']]}**. Selection uses the development lower confidence bound of incremental scenario value against no contact. These choices are recorded in `selection.json` before evaluation code accesses test outcomes.

For action a, the score is:

`Gamma_a = mu_a(X) + I(A=a) * (Y - mu_a(X)) / p_a`, with `p_a = 1/3`.

The score for policy pi is `margin * Gamma_pi - contact_cost * I(pi != 0)`. Differences between policies are calculated customer by customer, preserving their covariance. Randomization and out-of-sample nuisance fits support an unbiased value estimate even with a misspecified outcome model, subject to the experiment assumptions. The learner does not reveal true individual treatment effects.

## Primary held-out results

Assumptions: 40% contribution margin, $0.05 per contact, no binding contact cap. These are scenario inputs; contribution is not an observed financial outcome.

| Policy | Incremental scenario value/customer ($) | Pointwise 95% interval | Contacted |
|---|---:|---|---:|
{table}

![Policy values and uncertainty](policy_comparison.svg)

The primary paired comparison of the learned candidate against the simple comparator is {diff['estimate']:.4f}, 95% CI [{diff['lower']:.4f}, {diff['upper']:.4f}]. The IPW-only sensitivity estimate is {primary['ipw_sensitivity']['estimate']:.4f}, 95% CI [{primary['ipw_sensitivity']['lower']:.4f}, {primary['ipw_sensitivity']['upper']:.4f}]. Neither estimator establishes superiority. A positive point estimate for any other exploratory policy is not grounds to switch the selected candidate after viewing test results.

## Why uncertainty matters here

{result['heavy_tail_diagnostics']['zero_spend_fraction']:.2%} of evaluation customers spent nothing. The 99.9th percentile of spend is ${result['heavy_tail_diagnostics']['evaluation_spend_quantiles']['0.999']:.2f}; the maximum is ${result['heavy_tail_diagnostics']['evaluation_spend_quantiles']['1.0']:.2f}. Sparse purchases and a heavy spending tail make an incremental policy comparison noisy. The raw outcome was not trimmed after seeing the results. The report includes a paired customer bootstrap and IPW sensitivity rather than presenting a fragile point estimate alone.

## Scenario exploration

The offline demonstration evaluates a predefined grid of margins, contact costs and capacity limits. Learned policies contact only customers with positive predicted incremental scenario value and rank them under the cap. Uniform policies use a fixed random priority under a cap. All action choices use covariates and fixed model outputs, not test outcomes.

These secondary comparisons are exploratory. Displayed intervals are pointwise, conditional on the fixed trained models; capped-batch comparisons are conditional on this customer batch's covariates. They are not simultaneous confidence bands and must not be used to select a new winner and call it confirmatory. Bootstrap intervals for the primary contrast do not account for retraining variability across new experiments.

## Checks and limitations

Schema, numerical finiteness, treatment levels, missingness and outcome consistency pass. Training covariate balance is recorded without using it to exclude inconvenient observations. Tests cover known score identities, randomized null/effect simulations, exclusion of assigned treatment and outcomes from predictors, capacity limits and paired variance. Feature preparation and nuisance fitting respect split/fold boundaries.

The experimental publication is the source of the randomization claim; its allocation system was not independently audited. The observations are old and from one business setting. There is no evidence of current Indian campaign profitability, individual causal ground truth, long-term retention effects or actual retailer costs. Repeated sampling of training data could produce a different policy. The raw dataset has no express redistribution license established here and is obtained separately.

## Reproduction and references

Run `PYTHONPATH=src python -m unittest discover -s tests -v`, then `OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=src python -m customer_policy.experiment` and `PYTHONPATH=src python -m customer_policy.reporting`. The pinned dependencies, data hashes, split hashes and selection artifacts accompany this report.

- [Hillstrom original challenge](https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html).
- [scikit-uplift dataset documentation](https://www.uplift-modeling.com/en/latest/api/datasets/fetch_hillstrom.html).
- [GRF policy-learning guide](https://grf-labs.github.io/grf/articles/policy_learning.html), methodological background for policy learning and doubly robust scores. This project implements a different fixed DR-learner and does not claim to reproduce GRF's algorithm.
- [scikit-learn histogram gradient boosting](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html).
'''
    (reports/'TECHNICAL_REPORT.md').write_text(report.replace('$-', '−$'))
    memo=f'''# Decision memo: customer contact policy

**Recommendation:** retain the development-selected uniform men's email campaign for the defined historical population and primary cost scenario. Personalized targeting has not established incremental superiority.

The study used a 64,000-customer randomized experiment and a 12,800-customer untouched evaluation set. It compared uniform campaigns, a transparent purchase-segment policy and a cross-fitted doubly robust learner. Policy choices were fixed using separate development data.

The learned candidate's estimated difference from the simple comparator was ${diff['estimate']:.3f} per customer; its 95% interval was [${diff['lower']:.3f}, ${diff['upper']:.3f}]. A paired 2,000-resample bootstrap and an IPW-only estimate reached the same qualitative conclusion. The candidate contacted {result['policies'][primary['candidate']]['contact_fraction']:.1%} of customers, but the reduced contact count did not establish superior scenario value.

These figures assume a 40% margin and $0.05 contact cost. Actual margin and contact costs were not provided. The observation is incremental spending; financial contribution is a scenario calculation. Most customers spent zero, so uncertainty is material.

The practical decision is to avoid paying for a more complex targeting system on the strength of an unproven uplift claim. A modern deployment would need a fresh experiment, actual operating costs and evidence that treatment effects transport to the current population. The accompanying scenario explorer illustrates how assumptions change the analysis; it does not create new confirmatory evidence.

Evidence: [technical report](TECHNICAL_REPORT.md), [machine-readable results](results.json), [frozen selection](selection.json), [data audit](data_audit.json).
'''
    (reports/'DECISION_MEMO.md').write_text(memo.replace('$-', '−$'))
    grid=pd.read_csv(reports/'scenario_grid.csv').to_dict('records')
    payload=json.dumps({'primary':primary,'grid':grid,'labels':LABELS},separators=(',',':'))
    template='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Causal Customer Decisions · Prasanna K</title><style>
:root{color-scheme:light;--ink:#14243a;--muted:#53657a;--teal:#087769;--line:#dce4ea;--paper:#f5f7fa}*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.55 system-ui,-apple-system,Segoe UI,sans-serif}main{max-width:1160px;margin:auto;padding:44px 28px 70px}header{border-top:5px solid var(--teal);padding-top:24px}small,.eyebrow{font-size:12px;text-transform:uppercase;letter-spacing:1.5px;color:var(--muted)}h1{font:600 clamp(30px,4vw,48px)/1.12 Georgia,serif;margin:12px 0}h2{font-size:21px;margin:0 0 14px}p{max-width:900px}a{color:var(--teal);text-underline-offset:3px}.intro{font-size:19px;max-width:850px;color:var(--muted)}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:15px;margin:28px 0}.card,section{background:white;border:1px solid var(--line);border-radius:10px;padding:24px}.big{display:block;font-size:31px;letter-spacing:-1px;margin:6px 0}.card p{font-size:13px;margin:0;color:var(--muted)}section{margin-top:20px}.recommendation{border-left:5px solid var(--teal)}.tag{display:inline-block;border:1px solid var(--line);border-radius:20px;padding:4px 12px;font-size:12px;background:#f6f9fa}.controls{display:flex;gap:24px;flex-wrap:wrap;padding:16px 0 20px}label{display:grid;gap:6px;font-size:13px;font-weight:600}select{padding:10px 34px 10px 12px;border:1px solid #b6c5cf;border-radius:6px;background:white;color:var(--ink);font:inherit;min-width:160px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:12px 10px;border-bottom:1px solid var(--line)}th{font-size:12px;color:var(--muted)}td:nth-child(n+2),th:nth-child(n+2){text-align:right;font-variant-numeric:tabular-nums}.table-wrap{overflow-x:auto}.note{font-size:13px;color:var(--muted)}.row-primary{background:#eef8f5}.chart{width:100%;min-width:630px;height:250px}details{padding:14px 0;border-top:1px solid var(--line)}summary{cursor:pointer;font-weight:600}code{background:#edf1f5;padding:2px 6px;border-radius:4px;font-size:13px;overflow-wrap:anywhere}footer{margin-top:25px;color:var(--muted);font-size:13px}.sr{position:absolute;left:-9999px}@media(max-width:700px){main{padding:25px 16px}.cards{grid-template-columns:1fr}.card{padding:18px}.controls{gap:12px}section{padding:18px}.big{font-size:27px}}
</style></head><body><main><header><div class="eyebrow">Prasanna K · Experimental decision analysis</div><h1>When should a business personalize?</h1><p class="intro">A randomized customer experiment connects causal estimates to a contact decision—and tests whether added complexity earns its place.</p><span class="tag">64,000 customers</span> <span class="tag">Three randomized arms</span> <span class="tag">12,800 held-out customers</span></header><div class="cards"><div class="card"><small>Primary paired difference</small><strong class="big" id="difference"></strong><p>Learned policy minus simple campaign, per customer.</p></div><div class="card"><small>95% confidence interval</small><strong class="big" id="interval" style="font-size:25px"></strong><p>Includes zero; superiority is not established.</p></div><div class="card"><small>Decision</small><strong class="big" style="font-size:25px">Retain the simple policy</strong><p>For this population and the predefined cost scenario.</p></div></div><section class="recommendation"><h2>What the experiment supports</h2><p>The learned candidate did not establish an improvement over the uniform men's email campaign. Selection was frozen before evaluating the final sample. The study preserves that result instead of optimizing against the test set.</p><p class="note">“Men's” and “women's” describe campaign content. Customer gender is not observed. Financial contribution uses assumed margin and contact cost; it is not a measured profit figure.</p></section><section><div class="eyebrow">Secondary analysis · exploratory</div><h2>Inspect the decision assumptions</h2><p class="note">Choose a predefined scenario. These estimates reuse the evaluation sample; browsing them does not establish a new confirmatory winner. Capacity limits are evaluated for this fixed customer batch.</p><div class="controls"><label>Contribution margin<select id="margin"><option value="0.2">20%</option><option value="0.4" selected>40%</option><option value="0.6">60%</option></select></label><label>Cost per contact<select id="cost"><option value="0">$0.00</option><option value="0.05" selected>$0.05</option><option value="0.1">$0.10</option><option value="0.25">$0.25</option><option value="0.5">$0.50</option><option value="1">$1.00</option></select></label><label>Maximum contact share<select id="capacity"><option value="0.1">10%</option><option value="0.25">25%</option><option value="0.5">50%</option><option value="1" selected>100%</option></select></label></div><div class="table-wrap"><svg class="chart" id="chart" viewBox="0 0 920 250" role="img" aria-label="Point estimates and pointwise confidence intervals for incremental scenario value"></svg><table><thead><tr><th scope="col">Policy</th><th scope="col">Value/customer</th><th scope="col">Pointwise 95% interval</th><th scope="col">Contacted</th></tr></thead><tbody id="rows"></tbody></table></div><p id="scenario-note" class="note" aria-live="polite"></p></section><section><h2>How to challenge the result</h2><details open><summary>Inspect the experiment and evaluation</summary><p>Training, development and evaluation sets are separate. Assignment probabilities are known. A cross-fitted doubly robust learner competes with uniform campaigns and a transparent purchase-segment rule. The primary policy contrast uses paired customer scores and a 2,000-resample bootstrap.</p></details><details><summary>Understand the score</summary><p><code>Gamma_a = mu_a(X) + I(A=a) × (Y − mu_a(X)) / p_a</code></p><p>For the chosen action, scenario value is margin times the estimated spending score, less contact cost. Correct randomization supports value estimation even if the spending regression is imperfect.</p></details><details><summary>Check uncertainty and transport limits</summary><p>99.14% of the evaluation customers spent nothing. Sparse purchases and heavy tails limit precision. Intervals condition on the trained policies; they do not include variation from retraining on another experiment. Historical results do not establish current Indian campaign performance. The scenario grid is exploratory, with pointwise intervals rather than simultaneous confidence bands.</p></details><p><a href="https://github.com/Prasanna-K-123/causal-customer-decisions/blob/main/reports/TECHNICAL_REPORT.md">Technical report</a> · <a href="https://github.com/Prasanna-K-123/causal-customer-decisions/blob/main/reports/DECISION_MEMO.md">Decision memo</a> · <a href="https://github.com/Prasanna-K-123/causal-customer-decisions/blob/main/docs/PROTOCOL.md">Study protocol</a> · <a href="https://github.com/Prasanna-K-123/causal-customer-decisions/blob/main/docs/DATA.md">Data provenance</a></p><p class="note">Download this page to use the scenario explorer offline. All displayed values are saved aggregate results; no customer-level data or external service is used.</p></section><footer>Study date: 15 October 2026. Original data: Kevin Hillstrom / MineThatData. This is an empirical methods study, not a claim of realized commercial savings.</footer></main><script id="evidence" type="application/json">__PAYLOAD__</script><script>
const E=JSON.parse(document.getElementById('evidence').textContent);const money=x=>(x<0?'−':'')+'$'+Math.abs(x).toFixed(3);const P=E.primary.paired_difference;document.getElementById('difference').textContent=money(P.estimate);document.getElementById('interval').textContent='['+money(P.lower)+', '+money(P.upper)+']';
function render(){let m=+document.getElementById('margin').value,c=+document.getElementById('cost').value,k=+document.getElementById('capacity').value;let rows=E.grid.filter(r=>Math.abs(r.margin-m)<1e-8&&Math.abs(r.cost-c)<1e-8&&Math.abs(r.capacity-k)<1e-8);document.getElementById('rows').innerHTML=rows.map(r=>'<tr class="'+(r.policy===E.primary.comparator?'row-primary':'')+'"><th scope="row">'+E.labels[r.policy]+'</th><td>'+money(r.estimate)+'</td><td>['+money(r.lower)+', '+money(r.upper)+']</td><td>'+(100*r.contact_fraction).toFixed(1)+'%</td></tr>').join('');let lo=Math.min(0,...rows.map(r=>r.lower)),hi=Math.max(0,...rows.map(r=>r.upper));let pad=(hi-lo)*.10||.1;lo-=pad;hi+=pad;const x=v=>260+(v-lo)/(hi-lo)*610;let svg='<line x1="'+x(0)+'" y1="5" x2="'+x(0)+'" y2="205" stroke="#b5c2cb"/>';rows.forEach((r,i)=>{let y=22+i*38,col=r.policy===E.primary.comparator?'#087769':'#53657a';svg+='<text x="0" y="'+(y+5)+'" font-size="13" fill="#14243a">'+E.labels[r.policy]+'</text><line x1="'+x(r.lower)+'" y1="'+y+'" x2="'+x(r.upper)+'" y2="'+y+'" stroke="'+col+'" stroke-width="3"/><circle cx="'+x(r.estimate)+'" cy="'+y+'" r="5" fill="'+col+'"/>';});[lo,(lo+hi)/2,hi].forEach(v=>svg+='<text x="'+x(v)+'" y="230" text-anchor="middle" font-size="12" fill="#53657a">'+money(v)+'</text>');document.getElementById('chart').innerHTML=svg;document.getElementById('scenario-note').textContent=(m===.4&&c===.05&&k===1?'Primary scenario shown. ':'Exploratory scenario shown. ')+'All values are incremental to no contact. The shaded row is the original development-selected simple comparator; changing controls does not change the predefined primary comparison.';}
['margin','cost','capacity'].forEach(id=>document.getElementById(id).addEventListener('change',render));render();
</script></body></html>'''
    # Embed only controlled, aggregated numeric data and fixed labels.
    (docs/'index.html').write_text(template.replace('__PAYLOAD__',payload.replace('</','<\\/')))
    print('Generated technical report, decision memo, plot and offline scenario explorer.')

if __name__=='__main__':render(Path(__file__).resolve().parents[2])
