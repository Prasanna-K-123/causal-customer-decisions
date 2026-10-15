"""One reproducible training/development/locked-test run; saves aggregate evidence."""
from __future__ import annotations
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
import sklearn
from .data import acquire,load_checked,split_indices,treatment,audit,FEATURES,ARM_NAMES
from .evaluation import augmented_scores,policy_scores,summarize,paired_bootstrap,allocate,uniform_policy
from .models import DecisionModels


def dump(path:Path,value) -> None:
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")


def actions_for(name, predictions, row_ids, priority, margin,cost,capacity):
    if name in ["none","all_mens","all_womens"]:
        arm={"none":0,"all_mens":1,"all_womens":2}[name]
        return uniform_policy(arm,len(row_ids),capacity,priority)
    return allocate(predictions[name],margin,cost,capacity,row_ids)


def evaluate(frame,predictions,row_ids,priority,margin,cost,capacity,names):
    a=treatment(frame);y=frame.spend.to_numpy(float)
    gamma=augmented_scores(y,a,predictions['nuisance'])
    ipw=augmented_scores(y,a,np.zeros((len(frame),3)))
    reference=policy_scores(gamma,np.zeros(len(frame),int),margin,cost)
    results={};scores={};ipw_scores={};actions={}
    for name in names:
        act=actions_for(name,predictions,row_ids,priority,margin,cost,capacity)
        score=policy_scores(gamma,act,margin,cost)
        ipw_score=policy_scores(ipw,act,margin,cost)
        results[name]={"scenario_value":summarize(score),"incremental_vs_none":summarize(score-reference),"contact_fraction":float((act!=0).mean()),"action_counts":{ARM_NAMES[k]:int((act==k).sum()) for k in range(3)}}
        scores[name]=score;ipw_scores[name]=ipw_score;actions[name]=act
    return results,scores,ipw_scores,actions,gamma


def run(root:Path) -> dict:
    config=json.loads((root/'config.json').read_text());seed=config['seed']
    source=acquire(root);frame=load_checked(source)
    splits=split_indices(len(frame),seed)
    report=root/'reports';report.mkdir(exist_ok=True)
    dump(report/'data_audit.json',audit(frame,source,splits))
    # Only training outcomes enter fit; development and test are held separately.
    train=frame.iloc[splits['train']].reset_index(drop=True)
    dev=frame.iloc[splits['development']].reset_index(drop=True)
    test=frame.iloc[splits['test']].reset_index(drop=True)
    print('Fitting fixed models on',len(train),'training customers.',flush=True)
    models=DecisionModels(seed).fit(train)
    dev_pred=models.predict(dev)
    all_priority=np.random.default_rng(seed+10).random(len(frame))
    names=['none','all_mens','all_womens','segment','dr_learner']
    margin,cost,capacity=(config[k] for k in ['primary_margin','primary_contact_cost','primary_capacity'])
    dev_results,*_=evaluate(dev,dev_pred,splits['development'],all_priority[splits['development']],margin,cost,capacity,names)
    key=lambda name:dev_results[name]['incremental_vs_none']['lower']
    simple=max(names[:3],key=key);candidate=max(names[3:],key=key)
    selection={'seed':seed,'training_rows':len(train),'development_rows':len(dev),'simple_policy':simple,'candidate_policy':candidate,'rule':'Highest development lower 95% bound versus no contact within simple/learned families; held-out comparison fixes both choices.','primary_scenario':{'margin':margin,'contact_cost':cost,'capacity':capacity},'development_results':dev_results,'config_sha256':hashlib.sha256((root/'config.json').read_bytes()).hexdigest(),'protocol_sha256':hashlib.sha256((root/'docs/PROTOCOL.md').read_bytes()).hexdigest()}
    # Selection artifact is written before test outcomes enter evaluation.
    dump(report/'selection.json',selection)
    print('Selection frozen:',candidate,'versus',simple,flush=True)
    test_pred=models.predict(test)
    results,scores,ipw,actions,gamma=evaluate(test,test_pred,splits['test'],all_priority[splits['test']],margin,cost,capacity,names)
    paired=scores[candidate]-scores[simple]
    primary={'candidate':candidate,'comparator':simple,'scenario':selection['primary_scenario'],'paired_difference':summarize(paired),'bootstrap':paired_bootstrap(paired,seed+20,config['bootstrap_repetitions']),'ipw_sensitivity':summarize(ipw[candidate]-ipw[simple]),'evaluation_rows':len(test)}
    # Conservative predeclared evidence standard; not a second selection on test.
    supported=primary['paired_difference']['lower']>0 and primary['bootstrap']['lower']>0
    primary['superiority_supported']=bool(supported)
    primary['recommendation']=candidate if supported else simple
    primary['recommendation_reason']='Candidate superiority supported under the specified scenario.' if supported else 'Candidate superiority not established; retain the development-selected simple policy.'
    primary['inference_limits']='Fixed trained policies; customer-level sampling approximation; no present-day transport or realized profit claim.'
    itt=[]
    for k in [1,2]:
        itt.append({'contrast':ARM_NAMES[k]+' versus No E-Mail','spend_ate':summarize(gamma[:,k]-gamma[:,0]),'ipw_sensitivity':summarize(augmented_scores(test.spend.to_numpy(),treatment(test),np.zeros_like(gamma))[:,k]-augmented_scores(test.spend.to_numpy(),treatment(test),np.zeros_like(gamma))[:,0])})
    exploratory=[]
    for m in [.2,.4,.6]:
        for c in [0,.05,.10,.25,.50,1.]:
            for cap in [.10,.25,.50,1.]:
                r,*_=evaluate(test,test_pred,splits['test'],all_priority[splits['test']],m,c,cap,names)
                for name in names:
                    estimate=r[name]['incremental_vs_none']
                    exploratory.append({'margin':m,'cost':c,'capacity':cap,'policy':name,'contact_fraction':r[name]['contact_fraction'],**estimate})
    pd.DataFrame(exploratory).to_csv(report/'scenario_grid.csv',index=False)
    # SQL-friendly aggregate data summary; no row-level data published.
    summary=test.groupby('segment').agg(customers=('spend','size'),mean_spend=('spend','mean'),spend_std=('spend','std'),conversion_rate=('conversion','mean'),visit_rate=('visit','mean')).reset_index()
    summary.to_csv(report/'test_arm_summary.csv',index=False)
    balance=[]
    train_x=pd.get_dummies(train[FEATURES],columns=['zip_code','channel'],dtype=float)
    ta=treatment(train)
    for col in train_x:
        for k in [1,2]:
            x0=train_x.loc[ta==0,col].to_numpy(float);x1=train_x.loc[ta==k,col].to_numpy(float)
            denom=np.sqrt((x0.var(ddof=1)+x1.var(ddof=1))/2)
            balance.append({'feature':col,'arm':ARM_NAMES[k],'standardized_mean_difference':float((x1.mean()-x0.mean())/denom) if denom else 0.})
    pd.DataFrame(balance).to_csv(report/'training_balance.csv',index=False)
    output={'primary':primary,'policies':results,'intention_to_treat':itt,'heavy_tail_diagnostics':{'evaluation_spend_quantiles':{str(q):float(test.spend.quantile(q)) for q in [.5,.9,.95,.99,.999,1.]},'zero_spend_fraction':float((test.spend==0).mean())},'environment':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'scikit_learn':sklearn.__version__}}
    dump(report/'results.json',output)
    print(json.dumps(primary,indent=2),flush=True)
    return output


if __name__=='__main__':
    run(Path(__file__).resolve().parents[2])
