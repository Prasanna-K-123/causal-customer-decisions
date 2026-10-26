"""One-way train -> development lock -> final evaluation; extension only."""
from __future__ import annotations
import argparse,datetime,json,pickle,traceback,unittest
from pathlib import Path
import numpy as np
from .core import assert_group_disjoint
from .data import PROJECT,CONFIG,FREEZE,verify_protocol
from .models import NAMES,fit_propensities,fit_models,predict_bundle
from .evaluation import (select_ranking,factual_metrics,evaluate_rankings,sensitivity,
                         propensity_diagnostics)
from .runtime import Measurement,sha256_file,dump_json
from .synthetic import run_synthetic

def run(scratch,report):
    scratch,report=Path(scratch),Path(report)
    verify_protocol()
    measure=Measurement(report/"model_runtime.json")
    with measure.stage("public_tests_before_real_model_evaluation"):
        suite=unittest.defaultTestLoader.discover(str(PROJECT/"tests"),pattern="test*.py")
        receipt=unittest.TextTestRunner(verbosity=2).run(suite)
        if not receipt.wasSuccessful():raise RuntimeError("Public checks failed; final evaluation prohibited.")
        dump_json(report/"test_receipt.json",{"tests_run":receipt.testsRun,"failures":0,"errors":0,
            "scope":"Original and extension tests, before final Criteo outcome evaluation",
            "protocol_sha256":FREEZE["protocol_sha256"]})
    run_synthetic(measure,report/"synthetic_checks.json")
    with measure.stage("load_verified_private_cohort"):
        audit=json.loads((report/"data_audit.json").read_text())
        if audit["protocol_sha256"]!=FREEZE["protocol_sha256"]:raise ValueError("Audit protocol mismatch.")
        if sha256_file(scratch/"cohort.npz")!=audit["cohort"]["cohort_file_sha256"]:
            raise ValueError("Private cohort checksum differs from validated audit.")
        data=np.load(scratch/"cohort.npz")
        x,labels,fp,split,fold=(data[k] for k in ["x","labels","fp","split","fold"])
        assert_group_disjoint(fp,split);assert_group_disjoint(fp,fold)
        train,dev,test=split==0,split==1,split==2
        xt,tt,ft,foldt=x[train],labels[train,0],fp[train],fold[train]
    prop=fit_propensities(xt,tt,ft,foldt,measure,"criteo")
    bundles={};selections={};dev_predictions={}
    with measure.stage("development_propensity_prediction"):
        ed_raw=prop["full"].predict_proba(x[dev])[:,1];ed=np.clip(ed_raw,.05,.95)
    for outcome,label_index in [("visit",2),("conversion",1)]:
        bundle=fit_models(xt,tt,labels[train,label_index],ft,foldt,prop,measure,outcome)
        bundles[outcome]=bundle
        with measure.stage(outcome+"_development_prediction_selection"):
            pred=predict_bundle(bundle,x[dev]);dev_predictions[outcome]=pred
            winner,metrics=select_ranking(labels[dev,label_index],labels[dev,0],fp[dev],
                                          pred,ed,CONFIG["budgets"])
            selections[outcome]={"selected":winner,"development_integrated_release_scores":metrics}
    with measure.stage("write_selection_lock_before_test_evaluation"):
        models_path=scratch/"frozen_models.pkl"
        models_path.write_bytes(pickle.dumps({"bundles":bundles,"propensity":prop},protocol=5))
        lock={"protocol_sha256":FREEZE["protocol_sha256"],"config_sha256":FREEZE["config_sha256"],
            "source_sha256":audit["source_sha256"],"cohort_file_sha256":audit["cohort"]["cohort_file_sha256"],
            "split_hashes":{k:v["sorted_fingerprint_sha256"] for k,v in audit["cohort"]["splits"].items()},
            "frozen_model_file_sha256":sha256_file(models_path),"outcomes":selections,
            "created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "implementation_sha256":{str(p.relative_to(PROJECT)).replace("\\","/"):sha256_file(p)
                for p in sorted((PROJECT/"src/criteo_release").glob("*.py"))},
            "final_touch_rule":"No model selection, tuning, scenario changes or refit after this lock."}
        dump_json(report/"selection_lock.json",lock)
    results={"protocol_sha256":FREEZE["protocol_sha256"],"primary_outcome":"visit",
        "selection_lock_sha256":sha256_file(report/"selection_lock.json"),
        "claim_boundary":"Empirical released-data benchmark; conditional sensitivity is not identified campaign uplift",
        "uncertainty":"Pointwise fingerprint-cluster sandwich and primary Gaussian multiplier, conditional on fixed fits and batch ranks",
        "outcomes":{}}
    with measure.stage("test_propensity_prediction"):
        et_raw=prop["full"].predict_proba(x[test])[:,1];et=np.clip(et_raw,.05,.95)
        results["test_propensity"]=propensity_diagnostics(labels[test,0],et_raw,float(tt.mean()))
    for outcome,label_index in [("visit",2),("conversion",1)]:
        with measure.stage(outcome+"_test_prediction"):
            pred=predict_bundle(bundles[outcome],x[test])
        with measure.stage(outcome+"_locked_test_metrics"):
            yy,te,fe=labels[test,label_index],labels[test,0],fp[test]
            winner=selections[outcome]["selected"]
            results["outcomes"][outcome]={"status":"primary" if outcome=="visit" else "secondary",
                "factual_response":factual_metrics(yy,te,pred),
                "ranking":evaluate_rankings(yy,te,fe,pred,et,CONFIG["budgets"],winner,CONFIG["bootstrap_draws"]),
                "selection_sensitivity":sensitivity(pred["m0"],pred["m1"],fe,
                                    CONFIG["retention_ratios"],CONFIG["costs"][outcome])}
            np.savez_compressed(scratch/f"frozen_test_predictions_{outcome}.npz",
                y=yy,t=te,fp=fe,e=et,m0=pred["m0"],m1=pred["m1"],
                **{k:pred["scores"][k] for k in NAMES},
                **{f"prob_{k}":v for k,v in pred["response"].items()})
        dump_json(report/"results.json",results)
    with measure.stage("deterministic_frozen_prediction_repeat"):
        receipt={}
        reload=pickle.loads((scratch/"frozen_models.pkl").read_bytes())
        for outcome in ["visit","conversion"]:
            again=predict_bundle(reload["bundles"][outcome],x[test])
            frozen=np.load(scratch/f"frozen_test_predictions_{outcome}.npz")
            checks={k:float(np.max(np.abs(again["scores"][k]-frozen[k]))) for k in NAMES}
            if max(checks.values())>1e-14:raise AssertionError("Frozen prediction repeat mismatch.")
            receipt[outcome]=checks
        dump_json(report/"determinism_check.json",{
            "frozen_prediction_max_absolute_difference":receipt,
            "small_refit_determinism":"Independent identical fits in test_criteo_release",
            "scope":"Repeated predictions and a small refit; not a second complete real-data refit."})
    return results

def main():
    p=argparse.ArgumentParser();p.add_argument("--scratch",required=True)
    p.add_argument("--report",default=str(PROJECT/"reports/criteo"));args=p.parse_args()
    try:run(args.scratch,args.report)
    except Exception as exc:
        dump_json(Path(args.scratch)/"experiment_failure.json",{
            "type":type(exc).__name__,"message":str(exc),"traceback":traceback.format_exc()})
        raise
if __name__=="__main__":main()
