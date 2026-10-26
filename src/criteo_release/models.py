"""Fixed-budget release-response models; ranking scores are not identified effects."""
from __future__ import annotations
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from .core import augmented_contrast, assert_group_disjoint

NAMES = ["pooled_logistic","t_learner","dr_learner"]

def logistic():
    return make_pipeline(StandardScaler(),LogisticRegression(C=1,solver="lbfgs",
                         max_iter=400,tol=1e-7,random_state=20261026))

def outcome_model():
    return HistGradientBoostingClassifier(max_iter=80,learning_rate=.05,max_leaf_nodes=15,
        min_samples_leaf=200,l2_regularization=10,max_bins=255,early_stopping=False,
        random_state=20261026)

def ranking_model():
    return HistGradientBoostingRegressor(max_iter=80,learning_rate=.03,max_leaf_nodes=7,
        min_samples_leaf=500,l2_regularization=20,max_bins=255,early_stopping=False,
        random_state=20261026)

def fit_propensities(x,t,fp,fold,measurement,prefix):
    assert_group_disjoint(fp,fold)
    with measurement.stage(prefix+"_full_propensity"):
        full=logistic().fit(x,t)
    oof=np.empty(len(t))
    models=[]
    for k in range(3):
        inside=fold!=k
        if not inside.any() or not (~inside).any():
            raise ValueError("Empty crossfit fold.")
        with measurement.stage(prefix+f"_propensity_fold_{k}"):
            model=logistic().fit(x[inside],t[inside])
            oof[~inside]=model.predict_proba(x[~inside])[:,1]
        models.append(model)
    return {"full":full,"oof":np.clip(oof,.05,.95),"oof_unclipped":oof,"fold_models":models}

def fit_models(x,t,y,fp,fold,propensity,measurement,prefix):
    if not all(set(np.unique(y[t==a]))=={0,1} for a in [0,1]):
        raise ValueError("Both outcome classes are required within each training arm.")
    with measurement.stage(prefix+"_pooled_logistic"):
        pooled=logistic().fit(np.column_stack([x,t]),y)
    t_models=[]
    for a in [0,1]:
        with measurement.stage(prefix+f"_t_learner_arm_{a}"):
            model=outcome_model().fit(x[t==a],y[t==a])
        t_models.append(model)
    cross=np.empty((len(y),2))
    for k in range(3):
        inside=fold!=k
        with measurement.stage(prefix+f"_dr_nuisance_fold_{k}"):
            for a in [0,1]:
                rows=inside & (t==a)
                if len(np.unique(y[rows]))!=2:
                    raise ValueError("Both outcome classes are required within each training nuisance arm.")
                model=outcome_model().fit(x[rows],y[rows])
                cross[~inside,a]=model.predict_proba(x[~inside])[:,1]
    target=augmented_contrast(y,t,cross[:,0],cross[:,1],propensity["oof"])
    with measurement.stage(prefix+"_dr_ranking"):
        dr=ranking_model().fit(x,target)
    return {"pooled":pooled,"t_models":t_models,"dr":dr,
            "train_arm_means":[float(y[t==a].mean()) for a in [0,1]],
            "training_target_summary":{"mean":float(target.mean()),"sd":float(target.std()),
                "min":float(target.min()),"max":float(target.max())}}

def predict_bundle(bundle,x):
    n=len(x)
    pooled=np.column_stack([bundle["pooled"].predict_proba(
            np.column_stack([x,np.full(n,a)]))[:,1] for a in [0,1]])
    tm=np.column_stack([m.predict_proba(x)[:,1] for m in bundle["t_models"]])
    return {"scores":{"pooled_logistic":pooled[:,1]-pooled[:,0],
                      "t_learner":tm[:,1]-tm[:,0],"dr_learner":bundle["dr"].predict(x)},
            "response":{"arm_constant":np.tile(bundle["train_arm_means"],(n,1)),
                        "pooled_logistic":pooled,"t_learner":tm},
            "m0":tm[:,0],"m1":tm[:,1]}
