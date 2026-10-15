"""Fixed-complexity nuisance regressions, a segment policy and a DR learner."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from .data import NUMERIC, CATEGORICAL, treatment
from .evaluation import augmented_scores


def encoder() -> ColumnTransformer:
    return ColumnTransformer([("numeric", "passthrough", NUMERIC), ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL)])


def regressor(seed: int, effect: bool = False) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(loss="squared_error", max_iter=100, learning_rate=.03 if effect else .05, max_leaf_nodes=4 if effect else 7, min_samples_leaf=500 if effect else 200, l2_regularization=20 if effect else 10, early_stopping=False, random_state=seed)


def arm_models(x: np.ndarray, y: np.ndarray, a: np.ndarray, seed: int) -> list:
    return [regressor(seed+k).fit(x[a==k], y[a==k]) for k in range(3)]


def predict_means(models: list, x: np.ndarray) -> np.ndarray:
    return np.maximum(np.column_stack([m.predict(x) for m in models]),0.)


class DecisionModels:
    """Training outcomes only; evaluation never fits or tunes this object."""
    def __init__(self, seed: int = 20261015):
        self.seed=seed

    def fit(self, train: pd.DataFrame) -> "DecisionModels":
        y=train.spend.to_numpy(float);a=treatment(train)
        out_of_fold=np.empty((len(train),3))
        for fold,(fit_idx,pred_idx) in enumerate(KFold(3,shuffle=True,random_state=self.seed).split(train)):
            enc=encoder()
            fit_x=enc.fit_transform(train.iloc[fit_idx])
            predict_x=enc.transform(train.iloc[pred_idx])
            models=arm_models(fit_x,y[fit_idx],a[fit_idx],self.seed+fold*10)
            out_of_fold[pred_idx]=predict_means(models,predict_x)
        scores=augmented_scores(y,a,out_of_fold)
        self.encoder=encoder()
        x=self.encoder.fit_transform(train)
        self.nuisance_models=arm_models(x,y,a,self.seed)
        self.effect_models=[regressor(self.seed+100+k,True).fit(x,scores[:,k]-scores[:,0]) for k in [1,2]]
        self.arm_mean=np.array([y[a==k].mean() for k in range(3)])
        self.segment_means={}
        group=2*train.mens.to_numpy(int)+train.womens.to_numpy(int)
        for g in range(4):
            values=[]
            for k in range(3):
                local=y[(group==g)&(a==k)]
                values.append((local.sum()+200*self.arm_mean[k])/(len(local)+200))
            self.segment_means[g]=values
        self.training_rows=len(train)
        return self

    def predict(self, frame: pd.DataFrame) -> dict[str,np.ndarray]:
        x=self.encoder.transform(frame)
        mu=predict_means(self.nuisance_models,x)
        effects=np.column_stack([np.zeros(len(frame))]+[m.predict(x) for m in self.effect_models])
        group=2*frame.mens.to_numpy(int)+frame.womens.to_numpy(int)
        segment=np.array([self.segment_means[int(g)] for g in group])
        return {"nuisance":mu,"dr_learner":effects,"segment":segment}
