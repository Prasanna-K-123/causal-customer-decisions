"""Held-out response metrics, release-functional ranking, and conditional scenarios."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import average_precision_score,brier_score_loss,log_loss,roc_auc_score
from .core import (augmented_contrast,inverse_weighted_contrast,budget_weights,
                   integrated_contribution,clustered_summary,multiplier_interval,original_response)
from .models import NAMES

def factual_metrics(y,t,pred):
    out={}
    for name,prob in pred["response"].items():
        p=prob[np.arange(len(y)),t.astype(int)]
        out[name]={}
        for label,keep in [("overall",np.ones(len(y),bool)),("control",t==0),("treated",t==1)]:
            yy,pp=y[keep],p[keep]
            out[name][label]={"rows":len(yy),"positive_events":int(yy.sum()),
                "log_loss":float(log_loss(yy,pp,labels=[0,1])),
                "brier":float(brier_score_loss(yy,pp)),
                "roc_auc":float(roc_auc_score(yy,pp)) if len(np.unique(yy))>1 else None,
                "average_precision":float(average_precision_score(yy,pp)) if yy.sum()>0 else None}
    return out

def select_ranking(y,t,fp,pred,e,budgets):
    psi=augmented_contrast(y,t,pred["m0"],pred["m1"],e)
    metrics={name:float(integrated_contribution(pred["scores"][name],fp,psi,budgets).mean())
             for name in NAMES}
    winner=NAMES[0]
    for name in NAMES[1:]:
        if metrics[name]>metrics[winner]+1e-12:winner=name
    return winner,metrics

def propensity_diagnostics(t,raw,train_mean):
    clipped=np.clip(raw,.05,.95)
    return {"interpretation":"Fitted released-sample P(T=1|X), not known randomization probability",
        "raw_min":float(raw.min()),"raw_max":float(raw.max()),
        "quantiles":{str(q):float(np.quantile(raw,q)) for q in [0,.01,.05,.5,.95,.99,1]},
        "clipped_rows":int((raw!=clipped).sum()),"rows":len(t),
        "roc_auc":float(roc_auc_score(t,raw)),
        "log_loss":float(log_loss(t,raw)),
        "constant_train_proportion":float(train_mean),
        "constant_log_loss":float(log_loss(t,np.full(len(t),train_mean))),
        "warning":"AUC near .5 does not establish exchangeability or remove outcome selection."}

def evaluate_rankings(y,t,fp,pred,e,budgets,winner,draws):
    psi=augmented_contrast(y,t,pred["m0"],pred["m1"],e)
    ipw=inverse_weighted_contrast(y,t,e)
    values={name:integrated_contribution(score,fp,psi,budgets)
            for name,score in pred["scores"].items()}
    rows={}
    for name,score in pred["scores"].items():
        curves=[]
        for b in budgets:
            a=budget_weights(score,fp,b)
            curves.append({"budget":b,"actual_fraction":float(a.mean()),
                "release_contrast_vs_no_allocation":clustered_summary(a*psi,fp),
                "excess_vs_same_budget_random":clustered_summary((a-b)*psi,fp),
                "paired_excess_vs_pooled":clustered_summary(
                    (a-budget_weights(pred["scores"]["pooled_logistic"],fp,b))*psi,fp),
                "ipw_excess_vs_random":clustered_summary((a-b)*ipw,fp)})
        a=budget_weights(score,fp,.2)
        rows[name]={"preselected_on_development":name==winner,
            "integrated_excess_release_response":clustered_summary(values[name],fp),
            "integrated_ipw_excess":clustered_summary(
                integrated_contribution(score,fp,ipw,budgets),fp),
            "paired_integrated_difference_vs_pooled":clustered_summary(
                values[name]-values["pooled_logistic"],fp),
            "top_20_percent_diagnostics":{"weighted_rows":float(a.sum()),
                "weighted_control_rows":float(a[t==0].sum()),
                "weighted_treated_rows":float(a[t==1].sum()),
                "weighted_control_events":float((a[t==0]*y[t==0]).sum()),
                "weighted_treated_events":float((a[t==1]*y[t==1]).sum())},
            "budget_curve":curves}
    difference=values[winner]-values["pooled_logistic"]
    return {"estimand":"Released-sample standardized conditional response contrast; no causal identification claim",
        "common_release_contrast":clustered_summary(psi,fp),"candidates":rows,
        "preselected_vs_pooled":{"selected":winner,"paired":clustered_summary(difference,fp),
                                "cluster_multiplier":multiplier_interval(difference,fp,draws=draws)}}

def sensitivity(m0,m1,fp,ratios,costs):
    out=[]
    for r0 in ratios:
        for r1 in ratios:
            p0,p1=original_response(m0,r0),original_response(m1,r1)
            tau=p1-p0
            for cost in costs:
                a=budget_weights(tau-cost,fp,.2,positive_only=True)
                out.append({"r0":r0,"r1":r1,"response_unit_cost":cost,
                    "mean_reconstructed_contrast":float(tau.mean()),
                    "positive_contrast_fraction":float((tau>0).mean()),
                    "selected_fraction":float(a.mean()),
                    "conditional_gross_response_gain":float((a*tau).mean()),
                    "conditional_net_response_gain":float((a*(tau-cost)).mean())})
    return {"interpretation":"Model-based conditional scenarios, averaged over released test X; no causal or sampling CI",
            "unidentified_parameters":"Arm-specific constant positive/negative retention ratios",
            "capacity":.2,"scenarios":out}
