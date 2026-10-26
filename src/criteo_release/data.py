"""Stream every released row; retain only a fixed feature-group hash cohort."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from .core import (FEATURES,COLUMNS,CATEGORICAL,features,validate_frame,fingerprint,
                   mix64,split_ids,digest_array,assert_group_disjoint)
from .runtime import Measurement,sha256_file,dump_json

PROJECT = Path(__file__).resolve().parents[2]
CONFIG = json.loads((PROJECT/"docs/criteo/config_v1.json").read_text())
FREEZE = json.loads((PROJECT/"docs/criteo/protocol_freeze.json").read_text())

def verify_protocol():
    for filename,key in [("PROTOCOL_V1.md","protocol_sha256"),("config_v1.json","config_sha256")]:
        if sha256_file(PROJECT/"docs/criteo"/filename) != FREEZE[key]:
            raise ValueError("Frozen protocol/config checksum changed.")

def verify_source(source, expected_sha=None, expected_bytes=None):
    expected_sha = CONFIG["source_sha256"] if expected_sha is None else expected_sha
    expected_bytes = CONFIG["source_bytes"] if expected_bytes is None else expected_bytes
    if Path(source).stat().st_size != expected_bytes:
        raise ValueError("Compressed source size mismatch; no replacement is permitted.")
    actual = sha256_file(source)
    if actual != expected_sha:
        raise ValueError("Compressed source SHA256 mismatch; no replacement is permitted.")
    return actual

def audit(source, scratch, report):
    verify_protocol()
    scratch,report=Path(scratch),Path(report)
    scratch.mkdir(parents=True,exist_ok=True); report.mkdir(parents=True,exist_ok=True)
    measure=Measurement(report/"audit_runtime.json")
    with measure.stage("compressed_source_verification"):
        source_hash=verify_source(source)
    n=0; sums=np.zeros(12); squares=np.zeros(12)
    mins=np.full(12,np.inf); maxs=np.full(12,-np.inf)
    categorical={k:set() for k in CATEGORICAL}
    counts={k:{"0":0,"1":0} for k in COLUMNS[12:]}
    joint=np.zeros((2,2,2,2),dtype=np.int64)
    fp_parts=[]; xs=[]; ys=[]; fps=[]; row_ids=[]
    with measure.stage("full_stream_audit_and_hash_sample"):
        for chunk in pd.read_csv(source,compression="gzip",chunksize=CONFIG["chunksize"],
                                 dtype=np.float64,float_precision="round_trip"):
            x=validate_frame(chunk)
            labels=chunk[COLUMNS[12:]].to_numpy(dtype=np.uint8)
            fp=fingerprint(x)
            fp_parts.append(fp)
            take=mix64(fp,CONFIG["cohort_salt"]) % CONFIG["cohort_modulus"] == 0
            xs.append(x[take]); ys.append(labels[take]); fps.append(fp[take])
            row_ids.append(np.arange(n,n+len(chunk),dtype=np.int64)[take])
            sums+=x.sum(axis=0); squares+=(x*x).sum(axis=0)
            mins=np.minimum(mins,x.min(axis=0));maxs=np.maximum(maxs,x.max(axis=0))
            for j,k in enumerate(COLUMNS[12:]):
                pos=int(labels[:,j].sum()); counts[k]["1"]+=pos;counts[k]["0"]+=len(chunk)-pos
            np.add.at(joint,tuple(labels[:,j] for j in range(4)),1)
            for k in CATEGORICAL:
                categorical[k].update(np.unique(x[:,FEATURES.index(k)]).tolist())
            n+=len(chunk)
            if n%2000000==0: print(json.dumps({"rows_validated":n}),flush=True)
    if n!=CONFIG["source_rows"]:
        raise ValueError(f"Expected {CONFIG['source_rows']} rows, observed {n}.")
    with measure.stage("fingerprint_audit_and_private_cohort_save"):
        all_fp=np.concatenate(fp_parts);del fp_parts
        unique,groupcounts=np.unique(all_fp,return_counts=True)
        fingerprint_audit={"unique_groups":len(unique),"duplicate_feature_hash_rows":n-len(unique),
            "groups_with_multiplicity":int((groupcounts>1).sum()),
            "maximum_group_size":int(groupcounts.max()),
            "sorted_full_fingerprint_hash":digest_array(np.sort(all_fp)),
            "collision_caveat":"64-bit feature fingerprint, not a verified person ID; full-source collisions not exhaustively compared"}
        del all_fp,unique,groupcounts
        x,y,fp,rid=np.concatenate(xs),np.concatenate(ys),np.concatenate(fps),np.concatenate(row_ids)
        split=split_ids(fp,CONFIG["split_salt"])
        fold=(mix64(fp,CONFIG["fold_salt"])%3).astype(np.int8)
        assert_group_disjoint(fp,split);assert_group_disjoint(fp,fold)
        unique_features=pd.DataFrame(x,columns=FEATURES).assign(fp=fp).drop_duplicates(FEATURES)
        collision=int(unique_features.fp.duplicated().sum())
        if collision: raise ValueError("Observed distinct analytical feature vectors with same fingerprint.")
        split_audit={}
        for i,name in enumerate(["train","development","test"]):
            keep=split==i
            split_audit[name]={"rows":int(keep.sum()),"fingerprint_groups":int(np.unique(fp[keep]).size),
                "sorted_fingerprint_sha256":digest_array(np.sort(fp[keep])),
                "row_id_sha256":digest_array(rid[keep]),"content_sha256":digest_array(np.column_stack([x[keep],y[keep]])),
                "treatment_counts":{str(t):int((y[keep,0]==t).sum()) for t in [0,1]},
                "event_counts":{outcome:{str(t):int(y[keep & (y[:,0]==t),COLUMNS[12:].index(outcome)].sum())
                    for t in [0,1]} for outcome in ["visit","conversion"]}}
        np.savez_compressed(scratch/"cohort.npz",x=x,labels=y,fp=fp,split=split,fold=fold,row_id=rid)
        result={"protocol_sha256":FREEZE["protocol_sha256"],"source_sha256":source_hash,
            "compressed_source_bytes":Path(source).stat().st_size,"rows":n,"columns":COLUMNS,
            "missing_cells":0,"nonfinite_cells":0,"labels_validated_before_integer_conversion":True,
            "binary_counts":counts,"feature_audit":{f:{"min":float(mins[j]),"max":float(maxs[j]),
                "mean":float(sums[j]/n),"population_sd":float(np.sqrt(max(0,squares[j]/n-(sums[j]/n)**2))),
                "exact_cardinality":len(categorical[f]) if f in categorical else None}
                for j,f in enumerate(FEATURES)},
            "diagnostics":{"conversion_without_visit":int(joint[:,1,0,:].sum()),
                "control_exposure":int(joint[0,:,:,1].sum())},
            "full_fingerprint_audit":fingerprint_audit,
            "cohort":{"rows":len(x),"fraction_of_source":len(x)/n,
                "feature_fingerprint_collisions_observed":collision,
                "cross_partition_group_overlap":0,"crossfit_group_overlap":0,
                "cohort_file_sha256":sha256_file(scratch/"cohort.npz"),"splits":split_audit,
                "caveat":"Feature group is not a verified person/advertiser identity."}}
        dump_json(report/"data_audit.json",result)
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",required=True);p.add_argument("--scratch",required=True)
    p.add_argument("--report",default=str(PROJECT/"reports/criteo"))
    args=p.parse_args()
    try:audit(args.source,args.scratch,args.report)
    except Exception as exc:
        dump_json(Path(args.scratch)/"audit_failure.json",{"type":type(exc).__name__,"message":str(exc)})
        raise
if __name__=="__main__":main()
