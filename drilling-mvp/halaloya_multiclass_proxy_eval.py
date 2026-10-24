#!/usr/bin/env python3
"""Exploratory 5-class grouped-proxy evaluation for HalaLoya.

This is recorded-label diagnosis, NOT early warning. Depth-reset segments are
unverified proxy groups, NOT verified physical Well IDs. Explicit opt-in is required.

Example:
python halaloya_multiclass_proxy_eval.py data.xlsx --allow-depth-reset-proxy \
  --expected-critical-count 304 --drop 50 --max-iter 30 --bootstrap-reps 1000 \
  --out-json multiclass_eval.json --predictions-csv oof.csv --confusion-csv cm.csv
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from sklearn.metrics import cohen_kappa_score, mean_absolute_error

CLASSES = ["No-Loss", "Seepage Loss", "Partial Loss", "Severe Loss", "Complete Loss"]

def norm(s):
    return "".join(ch.lower() for ch in str(s) if ch.isalnum())

def get_col(df, wanted):
    matches = [c for c in df.columns if norm(c) == wanted]
    if len(matches) != 1:
        raise SystemExit(f"Expected one column matching {wanted!r}; found {matches}")
    return matches[0]

def load(path, sheet):
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"File not found: {p}")
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    if p.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
        return pd.read_excel(p, sheet_name=sheet if sheet is not None else 0)
    raise SystemExit("Input must be CSV or Excel.")

def cm_metrics(cm):
    tp = np.diag(cm).astype(float)
    true_support = cm.sum(axis=1).astype(float)
    pred_support = cm.sum(axis=0).astype(float)
    precision = np.divide(tp, pred_support, out=np.zeros(5), where=pred_support > 0)
    recall = np.divide(tp, true_support, out=np.zeros(5), where=true_support > 0)
    f1 = np.divide(2*precision*recall, precision+recall, out=np.zeros(5), where=(precision+recall)>0)
    total = int(cm.sum())
    obs = cm / total
    expected = np.outer(true_support/total, pred_support/total)
    weights = np.fromfunction(lambda i,j: ((i-j)**2)/16.0, (5,5))
    denom = float((weights*expected).sum())
    qwk = float(1-(weights*obs).sum()/denom) if denom > 0 else None
    return {
        "accuracy": float(tp.sum()/total),
        "balanced_accuracy": float(np.mean(recall[true_support > 0])),
        "macro_f1_all_5_classes": float(f1.mean()),
        "weighted_f1": float(np.dot(f1, true_support)/total),
        "quadratic_weighted_kappa": qwk,
        "ordinal_mae": float(sum(abs(i-j)*cm[i,j] for i in range(5) for j in range(5))/total),
        "per_class": {
            c: {"precision":float(precision[i]),"recall":float(recall[i]),"f1":float(f1[i]),
                "support":int(true_support[i]),"predicted_count":int(pred_support[i])}
            for i,c in enumerate(CLASSES)
        },
        "confusion_matrix": cm.astype(int).tolist()
    }

def ci(vals):
    x = np.asarray(vals, dtype=float); x = x[np.isfinite(x)]
    if not len(x):
        return {"valid_replicates":0,"ci_2_5":None,"median":None,"ci_97_5":None}
    lo, md, hi = np.percentile(x,[2.5,50,97.5])
    return {"valid_replicates":int(len(x)),"ci_2_5":float(lo),"median":float(md),"ci_97_5":float(hi)}

def group_bootstrap(group_cms, n, seed):
    rng = np.random.default_rng(seed)
    metric_values = {k:[] for k in ["accuracy","balanced_accuracy","macro_f1_all_5_classes",
                    "weighted_f1","quadratic_weighted_kappa","ordinal_mae"]}
    class_values = {c:{m:[] for m in ["precision","recall","f1"]} for c in CLASSES}
    absent = {c:0 for c in CLASSES}
    for _ in range(n):
        ix = rng.integers(0,len(group_cms),size=len(group_cms))
        cm = group_cms[ix].sum(axis=0)
        m = cm_metrics(cm)
        for k in metric_values:
            metric_values[k].append(m[k])
        for c in CLASSES:
            row = m["per_class"][c]
            if row["support"] == 0:
                absent[c] += 1
            else:
                for metric in ["precision","recall","f1"]:
                    class_values[c][metric].append(row[metric])
    return {
        "requested_replicates":n,
        "metrics":{k:ci(v) for k,v in metric_values.items()},
        "per_class":{c:{m:ci(v) for m,v in vals.items()} for c,vals in class_values.items()},
        "replicates_without_true_class_support":absent,
        "warning":"Cluster bootstrap is over inferred proxy segments and fixed out-of-segment predictions; it does not include model-fit uncertainty and segments are not verified wells."
    }

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path")
    ap.add_argument("--sheet",default=None)
    ap.add_argument("--allow-depth-reset-proxy",action="store_true")
    ap.add_argument("--drop",type=float,default=50)
    ap.add_argument("--expected-critical-count",type=int,default=None)
    ap.add_argument("--max-iter",type=int,default=30)
    ap.add_argument("--bootstrap-reps",type=int,default=500)
    ap.add_argument("--seed",type=int,default=0)
    ap.add_argument("--out-json",default="multiclass_proxy_eval.json")
    ap.add_argument("--predictions-csv",default="multiclass_oof_predictions.csv")
    ap.add_argument("--confusion-csv",default="multiclass_confusion_matrix.csv")
    args=ap.parse_args()
    if not args.allow_depth_reset_proxy:
        raise SystemExit("Pass --allow-depth-reset-proxy explicitly. These are proxy groups, not Well IDs.")
    df=load(args.path,args.sheet).reset_index(drop=True)
    target=get_col(df,"lossesseverity"); depth_col=get_col(df,"mdepth")
    if df[target].isna().any():
        raise SystemExit("Target has missing values; review first.")
    depth=pd.to_numeric(df[depth_col],errors="coerce")
    if depth.isna().any():
        raise SystemExit("Depth has missing/non-numeric values; review first.")
    observed=set(df[target].astype(str).unique())
    if observed != set(CLASSES):
        raise SystemExit(f"Unexpected label set {observed}; confirm the source label map before running.")
    y=pd.Categorical(df[target].astype(str),categories=CLASSES,ordered=True).codes.astype(int)
    critical=int(np.isin(y,[3,4]).sum())
    if args.expected_critical_count is not None and critical != args.expected_critical_count:
        raise SystemExit(f"Label guard failed: found {critical} Severe/Complete rows; expected {args.expected_critical_count}.")
    depth=depth.to_numpy(float)
    groups=np.r_[0,np.cumsum(np.diff(depth)<-args.drop)].astype(int)
    features=[c for c in df.select_dtypes(include="number").columns if c!=target and norm(c)!="mdepth"]
    if len(features)<2:
        raise SystemExit("Too few numeric predictors after excluding target and M.Depth.")
    X=df[features].replace([np.inf,-np.inf],np.nan).to_numpy(float)
    pred=np.full(len(df),-1,dtype=int); base=np.full(len(df),-1,dtype=int)
    fold_details=[]; cms=[]; segment_details=[]
    unique_groups=np.unique(groups)
    for fold,g in enumerate(unique_groups,1):
        test=groups==g; train=~test
        model=HistGradientBoostingClassifier(max_depth=4,learning_rate=.05,max_iter=args.max_iter,
            class_weight="balanced",random_state=args.seed,early_stopping=False)
        model.fit(X[train],y[train])
        pred[test]=model.predict(X[test])
        base[test]=int(np.bincount(y[train],minlength=5).argmax())
        cm=confusion_matrix(y[test],pred[test],labels=np.arange(5)); cms.append(cm)
        idx=np.flatnonzero(test)
        fold_details.append({"proxy_segment":int(g),"test_rows":int(test.sum()),
            "n_iter":int(model.n_iter_),
            "support":{c:int((y[test]==i).sum()) for i,c in enumerate(CLASSES)},
            "predicted":{c:int((pred[test]==i).sum()) for i,c in enumerate(CLASSES)}})
        segment_details.append({"proxy_segment":int(g),"first_row_1based":int(idx[0]+1),
            "last_row_1based":int(idx[-1]+1),"rows":int(len(idx)),
            "start_depth_m":float(depth[idx[0]]),"end_depth_m":float(depth[idx[-1]]),
            "critical_rows":int(np.isin(y[idx],[3,4]).sum()),"complete_loss_rows":int((y[idx]==4).sum())})
        print(f"Fold {fold}/{len(unique_groups)}: proxy segment {g}, rows={int(test.sum())}",flush=True)
    if (pred<0).any() or (base<0).any():
        raise RuntimeError("Some rows did not get out-of-proxy-segment predictions.")
    model_metrics=cm_metrics(confusion_matrix(y,pred,labels=np.arange(5)))
    baseline_metrics=cm_metrics(confusion_matrix(y,base,labels=np.arange(5)))
    report={
        "task":"Five-class classification of recorded LossesSeverity; not early warning",
        "input_file":Path(args.path).name,"rows":int(len(df)),"columns":int(len(df.columns)),
        "missing_cells":int(df.isna().sum().sum()),"exact_duplicates":int(df.duplicated().sum()),
        "target_counts":{c:int((y==i).sum()) for i,c in enumerate(CLASSES)},
        "critical_rows":critical,"critical_prevalence":float(critical/len(df)),
        "features":list(map(str,features)),"depth_excluded":True,
        "grouping":{"method":f"depth reset greater than {args.drop}m","proxy_group_count":int(len(unique_groups)),
            "actual_well_ids_available":False,"warning":"Leave-one-proxy-segment-out, NOT leave-one-well-out."},
        "model":{"type":"HistGradientBoostingClassifier","max_depth":4,"learning_rate":.05,
            "max_iter":args.max_iter,"class_weight":"balanced","early_stopping":False,"random_state":args.seed},
        "model_metrics":model_metrics,"fold_train_majority_baseline":baseline_metrics,
        "bootstrap":group_bootstrap(np.asarray(cms),args.bootstrap_reps,args.seed+2026),
        "folds":fold_details,"proxy_segments":segment_details,
        "limitations":["Inferred depth-reset segments are not verified Well IDs.",
            "Target is recorded severity on the same row, not future event onset.",
            "Complete Loss has only 34 rows and occurs in very few proxy segments.",
            "Bootstrap intervals condition on fixed predictions and do not include model-fitting uncertainty.",
            "The fixed iteration setting is exploratory and not tuned.",
            "No operational or safety use is supported."]
    }
    Path(args.out_json).write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
    pd.DataFrame({"row_1based":np.arange(1,len(df)+1),"proxy_segment":groups,"M.Depth":depth,
        "actual":np.asarray(CLASSES)[y],"predicted":np.asarray(CLASSES)[pred],
        "majority_baseline_predicted":np.asarray(CLASSES)[base]}).to_csv(args.predictions_csv,index=False)
    pd.DataFrame(confusion_matrix(y,pred,labels=np.arange(5)),
        index=[f"Actual: {c}" for c in CLASSES],columns=[f"Predicted: {c}" for c in CLASSES]).to_csv(args.confusion_csv)
    print(json.dumps({"model_metrics":model_metrics,"baseline":baseline_metrics,
        "bootstrap_metrics":report["bootstrap"]["metrics"],
        "class_absence_in_bootstrap":report["bootstrap"]["replicates_without_true_class_support"]},
        indent=2,ensure_ascii=False))
    print("Wrote",args.out_json,args.predictions_csv,args.confusion_csv)

if __name__=="__main__":
    main()
