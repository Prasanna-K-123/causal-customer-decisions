"""Independent aggregation from private frozen predictions and complete-rerun comparison."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from .runtime import dump_json,sha256_file
from .data import PROJECT,CONFIG
from .models import NAMES

def weights_independent(scores,groups,budget,positive=False):
    frame=pd.DataFrame({"group":groups,"score":scores})
    grouped=frame.groupby("group",sort=False).agg(score=("score","first"),size=("score","size"))
    grouped=grouped.sort_values(["score","group"],ascending=[False,True])
    remaining=float(budget*len(frame));allocation={}
    for row in grouped.itertuples():
        take=min(float(row.size),max(0,remaining)) if not positive or row.score>0 else 0.
        allocation[row.Index]=take/row.size
        remaining-=take
    return np.array([allocation[g] for g in groups])

def independent_summary(v,g):
    mean=float(np.mean(v));frame=pd.DataFrame({"g":g,"centered":v-mean})
    sums=frame.groupby("g").centered.sum().to_numpy()
    se=float(np.sqrt(len(sums)/(len(sums)-1)*np.sum(sums*sums)/len(v)**2))
    return mean,se

def verify(scratch,report,repeat=None):
    scratch,report=Path(scratch),Path(report)
    result=json.loads((report/"results.json").read_text())
    lock=json.loads((report/"selection_lock.json").read_text())
    checks=[]
    def compare(label,a,b):
        delta=float(np.max(np.abs(np.asarray(a,float)-np.asarray(b,float))))
        if delta>1e-10:raise AssertionError(f"{label}: difference {delta}")
        checks.append({"check":label,"max_absolute_difference":delta})
    for outcome in ["visit","conversion"]:
        d=np.load(scratch/f"frozen_test_predictions_{outcome}.npz")
        y,t,e,m0,m1,g=(d[k] for k in ["y","t","e","m0","m1","fp"])
        # Separate arithmetic expression; no benchmark statistical helper is imported.
        psi=np.where(t==1,(y-m1)/e,-(y-m0)/(1-e))+(m1-m0)
        contributions={}
        for name in NAMES:
            vals=[]
            for row in result["outcomes"][outcome]["ranking"]["candidates"][name]["budget_curve"]:
                b=row["budget"]
                a=weights_independent(d[name],g,b)
                v=(a-b)*psi
                estimate,se=independent_summary(v,g)
                ref=row["excess_vs_same_budget_random"]
                compare(outcome+"/"+name+f"/b={b}",[estimate,se],[ref["estimate"],ref["se"]])
                vals.append(v)
            # Manual trapezoids instead of np.trapezoid.
            area=np.zeros(len(y))
            for j in range(1,len(vals)):
                area+=(CONFIG["budgets"][j]-CONFIG["budgets"][j-1])*(vals[j]+vals[j-1])/2
            contributions[name]=area
            ref=result["outcomes"][outcome]["ranking"]["candidates"][name]["integrated_excess_release_response"]
            compare(outcome+"/"+name+"/integral",independent_summary(area,g),[ref["estimate"],ref["se"]])
        winner=lock["outcomes"][outcome]["selected"]
        ref=result["outcomes"][outcome]["ranking"]["preselected_vs_pooled"]["paired"]
        compare(outcome+"/paired_primary",independent_summary(
            contributions[winner]-contributions["pooled_logistic"],g),[ref["estimate"],ref["se"]])
        for name in ["arm_constant","pooled_logistic","t_learner"]:
            p=d["prob_"+name][np.arange(len(y)),t]
            eps=np.finfo(float).eps;p=np.clip(p,eps,1-eps)
            ll=-np.mean(y*np.log(p)+(1-y)*np.log1p(-p))
            br=np.mean((p-y)**2)
            ref=result["outcomes"][outcome]["factual_response"][name]["overall"]
            compare(outcome+"/"+name+"/response",[ll,br],[ref["log_loss"],ref["brier"]])
        for row in result["outcomes"][outcome]["selection_sensitivity"]["scenarios"]:
            # Odds division is algebraically independent of the primary Bayes expression.
            p0=1/(1+row["r0"]*(1-m0)/m0)
            p1=1/(1+row["r1"]*(1-m1)/m1)
            compare(outcome+f"/r0={row['r0']}/r1={row['r1']}/mean",(p1-p0).mean(),
                    row["mean_reconstructed_contrast"])
    receipt={"status":"passed","checks":checks,"number_of_checks":len(checks),
        "max_absolute_difference":max(c["max_absolute_difference"] for c in checks),
        "scope":"Independent group allocation, arithmetic scores, manual trapezoid, cluster SE, factual loss and odds inversion",
        "predictions_sha256":{o:sha256_file(scratch/f"frozen_test_predictions_{o}.npz")
            for o in ["visit","conversion"]}}
    if repeat:
        repeat=Path(repeat)
        rr=json.loads((repeat/"reports/results.json").read_text())
        differences=[]
        def walk(a,b,p=""):
            if isinstance(a,dict):
                if set(a)!=set(b):raise AssertionError(p+" key mismatch")
                for k in a:
                    if k!="selection_lock_sha256":walk(a[k],b[k],p+"/"+k)
            elif isinstance(a,list):
                if len(a)!=len(b):raise AssertionError(p+" length mismatch")
                for i,(aa,bb) in enumerate(zip(a,b)):walk(aa,bb,p+f"/{i}")
            elif isinstance(a,(float,int)) and not isinstance(a,bool):
                delta=abs(a-b)
                if delta>1e-10:raise AssertionError(f"{p} repeat differs: {delta}")
                differences.append(delta)
            elif a!=b:raise AssertionError(p+" value mismatch")
        walk(result,rr)
        syn=json.loads((report/"synthetic_checks.json").read_text())
        syn2=json.loads((repeat/"reports/synthetic_checks.json").read_text())
        walk(syn,syn2,"synthetic")
        receipt["complete_model_rerun"]={"status":"passed","numeric_fields_compared":len(differences),
            "max_absolute_difference":max(differences),"tolerance":1e-10,
            "same_frozen_model_sha256":sha256_file(scratch/"frozen_models.pkl")==sha256_file(repeat/"frozen_models.pkl"),
            "scope":"All fixed synthetic/real fits and evaluation rerun; verified cohort reused; no retuning"}
    dump_json(report/"verification.json",receipt)
    return receipt

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--scratch",required=True)
    p.add_argument("--report",default=str(PROJECT/"reports/criteo"));p.add_argument("--repeat")
    a=p.parse_args();r=verify(a.scratch,a.report,a.repeat)
    print(json.dumps({k:v for k,v in r.items() if k!="checks"},indent=2))
