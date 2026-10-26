"""Known-truth checks: exact selected distributions and independent sampled outcomes."""
from __future__ import annotations
import numpy as np
from .core import (fingerprint,mix64,synthetic_truth,selected_response,original_response,
                   augmented_contrast,budget_weights,clustered_summary)
from .models import fit_propensities,fit_models,predict_bundle
from .runtime import dump_json

def exact_selection_oracle():
    p0=np.array([.01,.04,.10]);p1=np.array([.02,.06,.08])
    weights=np.array([.3,.4,.3]);r0,r1=2,5
    s0,s1=selected_response(p0,r0),selected_response(p1,r1)
    recovered=original_response(s1,r1)-original_response(s0,r0)
    true=p1-p0
    np.testing.assert_allclose(recovered,true,rtol=0,atol=1e-15)
    return {"stratum_weights":weights.tolist(),"original_p0":p0.tolist(),"original_p1":p1.tolist(),
        "r0":r0,"r1":r1,"selected_s0":s0.tolist(),"selected_s1":s1.tolist(),
        "true_weighted_contrast":float(weights@true),
        "naive_selected_contrast_at_original_X_weights":float(weights@(s1-s0)),
        "oracle_recovered_contrast":float(weights@recovered),
        "max_absolute_recovery_error":float(np.max(np.abs(recovered-true))),
        "constant_X_counterexample":{"p0":.04,"p1":.06,"original_contrast":.02,
            "selected_contrast":float(selected_response(.06,5)-selected_response(.04,2)),
            "X_T_independent_before_and_after_selection":True},
        "warning":"Truth and retention parameters are known only because this is constructed."}

def run_synthetic(measurement,output):
    rng=np.random.default_rng(20261026)
    def sample(n):
        x=np.zeros((n,12));x[:,:4]=rng.uniform(-1,1,(n,4))
        p0,p1=synthetic_truth(x);t=rng.binomial(1,.5,n)
        y=rng.binomial(1,np.where(t==1,p1,p0))
        return x,t,y,p0,p1
    x,t,y,_,_=sample(60000)
    xe,te,ye,p0,p1=sample(40000)
    fp=fingerprint(x);fpe=fingerprint(xe);fold=(mix64(fp,2026102603)%3).astype(np.int8)
    prop=fit_propensities(x,t,fp,fold,measurement,"synthetic")
    fit=fit_models(x,t,y,fp,fold,prop,measurement,"synthetic")
    with measurement.stage("synthetic_prediction_and_known_truth"):
        pred=predict_bundle(fit,xe);tau=p1-p0
        psi=augmented_contrast(ye,te,pred["m0"],pred["m1"],np.full(len(ye),.5))
        result={"training_rows":60000,"evaluation_rows":40000,"seed":20261026,
            "known_treatment_probability":.5,"truth":exact_selection_oracle(),"models":{}}
        for name,score in pred["scores"].items():
            a=budget_weights(score,fpe,.2)
            true_gain=float(((a-.2)*tau).mean())
            summary=clustered_summary((a-.2)*psi,fpe)
            result["models"][name]={"pehe":float(np.sqrt(np.mean((score-tau)**2))),
                "true_top20_excess_vs_random":true_gain,"estimated_top20_excess_vs_random":summary,
                "truth_inside_pointwise_interval":summary["ci95"][0]<=true_gain<=summary["ci95"][1]}
        result["true_mean_effect"]=float(tau.mean())
        result["oracle_top20_excess_vs_random"]=float(
            ((budget_weights(tau,fpe,.2)-.2)*tau).mean())
        dump_json(output,result)
    return result
