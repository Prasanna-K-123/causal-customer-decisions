"""Plots and aggregate tables generated only from saved, locked results."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .data import PROJECT
from .runtime import Measurement

LABELS={"pooled_logistic":"Pooled logistic","t_learner":"Two-model boosting","dr_learner":"Cross-fitted DR ranking"}
COLORS={"pooled_logistic":"#526880","t_learner":"#087d76","dr_learner":"#c16931"}

def main():
    folder=PROJECT/"reports/criteo"
    r=json.loads((folder/"results.json").read_text())
    measure=Measurement(folder/"reporting_runtime.json")
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,
        "axes.spines.right":False,"axes.titleweight":"bold","svg.hashsalt":"criteo-v1"})
    with measure.stage("aggregate_tables_and_figures"):
        metrics=[];curves=[];scenarios=[]
        fig,axes=plt.subplots(1,2,figsize=(12,4.4),layout="constrained")
        for ax,(outcome,d) in zip(axes,r["outcomes"].items()):
            for name,entry in d["ranking"]["candidates"].items():
                rows=entry["budget_curve"];x=np.array([z["budget"] for z in rows])*100
                y=np.array([z["excess_vs_same_budget_random"]["estimate"] for z in rows])*100
                interval=np.array([z["excess_vs_same_budget_random"]["ci95"] for z in rows])*100
                ax.plot(x,y,"o-",label=LABELS[name],color=COLORS[name],markersize=3)
                if entry["preselected_on_development"]:
                    ax.fill_between(x,interval[:,0],interval[:,1],color=COLORS[name],alpha=.12,
                                    label="Selected model: pointwise 95% CI")
                for z in rows:
                    v=z["excess_vs_same_budget_random"]
                    curves.append({"outcome":outcome,"model":name,"budget":z["budget"],
                        "estimate":v["estimate"],"se":v["se"],"ci_lower":v["ci95"][0],"ci_upper":v["ci95"][1]})
            ax.axhline(0,color="#333333",linewidth=.7)
            ax.set(xlabel="Allocation budget (%)",ylabel="Excess release response (percentage points)",
                   title=outcome.capitalize()+(" · primary" if outcome=="visit" else " · secondary"))
            ax.grid(alpha=.15);ax.legend(fontsize=7,loc="lower right")
            for name,sets in d["factual_response"].items():
                for subset,values in sets.items():metrics.append({"outcome":outcome,"model":name,"subset":subset,**values})
            for row in d["selection_sensitivity"]["scenarios"]:scenarios.append({"outcome":outcome,**row})
        fig.suptitle("Held-out ranking of released responses — no original-campaign causal interpretation",fontsize=12)
        fig.savefig(folder/"release_budget_curves.svg",metadata={"Date":None});plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(11.5,4.5),layout="constrained")
        ratios=[1,2,5,10]
        for ax,(outcome,d) in zip(axes,r["outcomes"].items()):
            rows=[z for z in d["selection_sensitivity"]["scenarios"] if z["response_unit_cost"]==0]
            matrix=np.array([[next(z["mean_reconstructed_contrast"] for z in rows
                            if z["r0"]==r0 and z["r1"]==r1)*100 for r1 in ratios] for r0 in ratios])
            lim=np.max(np.abs(matrix))
            im=ax.imshow(matrix,cmap="BrBG",vmin=-lim,vmax=lim)
            for i in range(4):
                for j in range(4):
                    ax.text(j,i,f"{matrix[i,j]:+.3f}",ha="center",va="center",
                            color="white" if abs(matrix[i,j])>.65*lim else "#152932",fontsize=10)
            ax.set(xticks=range(4),xticklabels=ratios,yticks=range(4),yticklabels=ratios,
                xlabel="Treated-arm retention ratio r1",ylabel="Control-arm retention ratio r0",title=outcome.capitalize())
            fig.colorbar(im,ax=ax,shrink=.76,label="Conditional contrast (percentage points)")
        fig.suptitle("Sensitivity to assumed positive/negative retention — fixed released test covariates",fontsize=12)
        fig.savefig(folder/"selection_sensitivity.svg",metadata={"Date":None});plt.close(fig)
        audit=json.loads((folder/"audit_runtime.json").read_text())
        run=json.loads((folder/"model_runtime.json").read_text())
        groups={"Source hash + full audit":sum(z["wall_seconds"] for z in audit["stages"]),
            "Synthetic + checks":sum(z["wall_seconds"] for z in run["stages"] if z["stage"].startswith("synthetic") or z["stage"].startswith("public_tests")),
            "Criteo model training":sum(z["wall_seconds"] for z in run["stages"] if
                any(z["stage"].startswith(p) for p in ["criteo_","visit_","conversion_"]) and
                not any(q in z["stage"] for q in ["prediction","selection","metrics"])),
            "Prediction + evaluation":sum(z["wall_seconds"] for z in run["stages"] if
                any(q in z["stage"] for q in ["prediction","selection","metrics"]) and not z["stage"].startswith("synthetic"))}
        fig,ax=plt.subplots(figsize=(9.5,3.7),layout="constrained")
        bars=ax.barh(list(groups),list(groups.values()),color=["#526880","#c16931","#087d76","#6d6491"])
        ax.bar_label(bars,fmt="%.1f s",padding=4);ax.set_xlim(0,max(groups.values())*1.17)
        ax.set_xlabel("Wall seconds on the recorded shared local machine")
        ax.set_title("Measured workload: full audit, 524,458-row model training, 174,618-row test")
        ax.grid(axis="x",alpha=.15)
        fig.savefig(folder/"local_runtime.svg",metadata={"Date":None});plt.close(fig)
        pd.DataFrame(metrics).to_csv(folder/"factual_metrics.csv",index=False)
        pd.DataFrame(curves).to_csv(folder/"budget_curves.csv",index=False)
        pd.DataFrame(scenarios).to_csv(folder/"selection_scenarios.csv",index=False)
    print("Generated aggregate CSVs and three SVG figures from locked results.")
if __name__=="__main__":main()
