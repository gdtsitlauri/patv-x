"""
PATV-X+ Ablation Study — 4-Level (L1+L2+L3+L4/CTCG)
====================================================

Βελτιώσεις:
  1) Ablation για το L4 (CTCG) layer
  2) Per-CTCG-submodule ablation
  3) 4-D weight grid search (L1, L2, L3, L4)
  4) Bootstrap 95% CI για AUC
  5) 16-D features

Χρήση:
    python ablation_study.py --synthetic
    python ablation_study.py --data pipeline_results/features.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from itertools import product
from pathlib import Path

import numpy as np

DEFAULT_WEIGHTS = (0.14, 0.34, 0.02, 0.50)   # L1, L2, L3, L4 — v12 calibrated on FaceForensics++ ablation

FEATURE_NAMES = [
    "flow_score","mean_divergence","acceleration_anomaly","flicker_score",
    "physics_score","gravity_consistency","rigid_body_score","shadow_consistency",
    "semantic_score","color_drift","edge_stability","texture_consistency",
    "ctcg_score","ctcg_phase_coherence","ctcg_ar_residual","ctcg_spectral_anomaly",
]

METRIC_DISPLAY_NAMES = [
    "flow_score (L1)","mean_divergence (L1)","acceleration_anomaly (L1)","flicker_score (L1)",
    "physics_score (L2)","gravity_consistency (L2)","rigid_body_score (L2)","shadow_consistency (L2)",
    "semantic_score (L3)","color_drift (L3)","edge_stability (L3)","texture_consistency (L3)",
    "ctcg_score (L4)","ctcg_phase_coherence (L4)","ctcg_ar_residual (L4)","ctcg_spectral_anomaly (L4)",
]


def compute_auc(scores, labels):
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    np1 = int(labels.sum()); nn = len(labels)-np1
    if np1==0 or nn==0:
        return 0.5
    order = np.argsort(scores)
    s = scores[order]
    y = labels[order]
    neg_before = 0
    favorable_pairs = 0.0
    i = 0
    while i < len(s):
        j = i + 1
        while j < len(s) and s[j] == s[i]:
            j += 1
        group = y[i:j]
        pos = int(group.sum())
        neg = int(len(group) - pos)
        favorable_pairs += pos * neg_before + 0.5 * pos * neg
        neg_before += neg
        i = j
    return float(favorable_pairs / max(np1 * nn, 1))


def best_f1(scores, labels):
    bs,bt=-1.0,0.5
    for t in np.arange(0.05,0.96,0.01):
        p=(scores>=t).astype(int)
        tp=int(((p==1)&(labels==1)).sum()); fp=int(((p==1)&(labels==0)).sum())
        fn=int(((p==0)&(labels==1)).sum())
        prec=tp/(tp+fp+1e-9); rec=tp/(tp+fn+1e-9); f1=2*prec*rec/(prec+rec+1e-9)
        if f1>bs: bs,bt=float(f1),float(t)
    return bs,bt


def accuracy_at(scores,labels,thr):
    return float(((scores>=thr).astype(int)==labels).mean())


def bootstrap_auc_ci(scores, labels, n=300, seed=0):
    rng=np.random.default_rng(seed); aucs=[]
    for _ in range(n):
        idx=rng.integers(0,len(labels),len(labels))
        aucs.append(compute_auc(scores[idx],labels[idx]))
    aucs=sorted(aucs)
    return float(aucs[int(0.025*n)]), float(aucs[int(0.975*n)])


def load_csv(csv_path):
    X,y=[],[]
    with open(Path(csv_path),newline="",encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                feats=[
                    float(row.get("flow_score",row.get("L1_flow",0)) or 0),
                    float(row.get("mean_divergence",0) or 0),
                    float(row.get("acceleration_anomaly",0) or 0),
                    float(row.get("flicker_score",0) or 0),
                    float(row.get("physics_score",row.get("L2_physics",0)) or 0),
                    float(row.get("gravity_consistency",0) or 0),
                    float(row.get("rigid_body_score",0) or 0),
                    float(row.get("shadow_consistency",0) or 0),
                    float(row.get("semantic_score",row.get("L3_semantic",0)) or 0),
                    float(row.get("color_drift",0) or 0),
                    float(row.get("edge_stability",0) or 0),
                    float(row.get("texture_consistency",0) or 0),
                    float(row.get("ctcg_score",row.get("L4_ctcg",0)) or 0),
                    float(row.get("ctcg_phase_coherence",0) or 0),
                    float(row.get("ctcg_ar_residual",0) or 0),
                    float(row.get("ctcg_spectral_anomaly",0) or 0),
                ]
                lbl = int(float(row["label"])) if row.get("label","").strip() else (
                    1 if str(row.get("verdict","")).upper()=="AI_GENERATED" else 0)
                X.append(feats); y.append(lbl)
            except (ValueError,TypeError): continue
    if not X: raise ValueError("Δεν βρέθηκαν γραμμές.")
    return np.clip(np.nan_to_num(np.array(X,float),0,1,0),0,1), np.array(y,int)


def generate_synthetic_data(n=500, seed=42):
    # Calibrated from real FaceForensics++ ablation: L2 is primary discriminator
    rng=np.random.default_rng(seed); k=n//2
    am=[0.15,0.18,0.07,0.08, 0.28,0.22,0.18,0.10, 0.12,0.08,0.12,0.10, 0.15,0.12,0.14,0.11]
    as_=[0.06,0.08,0.04,0.04, 0.09,0.09,0.08,0.05, 0.06,0.05,0.05,0.05, 0.07,0.07,0.07,0.06]
    im=[0.38,0.22,0.17,0.16, 0.52,0.26,0.48,0.16, 0.19,0.13,0.14,0.26, 0.32,0.28,0.30,0.25]
    is_=[0.10,0.08,0.08,0.07, 0.11,0.09,0.11,0.07, 0.08,0.07,0.06,0.10, 0.11,0.11,0.11,0.10]
    Xa=np.clip(rng.normal(am,as_,(k,16)),0,1); Xi=np.clip(rng.normal(im,is_,(k,16)),0,1)
    X=np.vstack([Xa,Xi]); y=np.concatenate([np.zeros(k,int),np.ones(k,int)])
    idx=rng.permutation(len(X)); return X[idx],y[idx]


def detect_split_manifest(data_path, explicit_path=None):
    if explicit_path:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"Δεν βρέθηκε split manifest: {path}")
        return path
    if not data_path:
        return None
    candidate = Path(data_path).resolve().with_name("split_manifest.csv")
    return candidate if candidate.exists() else None


def dev_subset_from_manifest(X, y, manifest_path):
    keep = []
    with open(manifest_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            idx = int(float(row.get("index", -1)))
            split = str(row.get("split", "")).strip().lower()
            if idx >= 0 and split in {"train", "validation", "val"}:
                keep.append(idx)
    keep = np.array(sorted(set(i for i in keep if i < len(X))), dtype=int)
    if keep.size == 0:
        raise ValueError(f"Άδειο development subset στο manifest: {manifest_path}")
    return X[keep], y[keep], keep


def compute_tcs(X,w1,w2,w3,w4):
    return w1*X[:,0]+w2*X[:,4]+w3*X[:,8]+w4*X[:,12]


def compute_masked_tcs(X,mask=None,weights=DEFAULT_WEIGHTS):
    Xm=X.copy()
    if mask is not None: Xm[:,~mask]=0.0
    grp=[slice(0,4),slice(4,8),slice(8,12),slice(12,16)]
    sc=[]
    for (g,w) in zip(grp,weights):
        gv=Xm[:,g]; act=np.any(gv!=0,axis=1)
        gs=np.mean(gv,axis=1); gs[~act]=0.0
        sc.append(w*gs)
    return np.sum(sc,axis=0)


def run_level_ablation(X,y):
    print("\n  [Ablation 1] Level Contribution")
    print("  "+"─"*60)
    configs={
        "All (L1+L2+L3+L4)":   (0.18,0.42,0.02,0.38),
        "L1+L2+L3 (no CTCG)":  (0.30,0.65,0.05,0.00),
        "L1+L2+L4":             (0.18,0.42,0.00,0.40),
        "L2+L3+L4":             (0.00,0.50,0.10,0.40),
        "L1+L3+L4":             (0.30,0.00,0.30,0.40),
        "L4 only (CTCG)":       (0.00,0.00,0.00,1.00),
        "L1+L2 only":           (0.30,0.70,0.00,0.00),
        "L2 only":              (0.00,1.00,0.00,0.00),
        "L1 only":              (1.00,0.00,0.00,0.00),
        "L3 only":              (0.00,0.00,1.00,0.00),
    }
    results={}; base_auc=None
    for name,(w1,w2,w3,w4) in configs.items():
        sc=compute_tcs(X,w1,w2,w3,w4)
        auc=compute_auc(sc,y); f1,thr=best_f1(sc,y); acc=accuracy_at(sc,y,thr)
        lo,hi=bootstrap_auc_ci(sc,y,n=200)
        if base_auc is None: ds=""; mk="← baseline"; base_auc=auc
        else: ds=f"{base_auc-auc:+.4f}"; mk=""
        print(f"    {name:25s}  AUC={auc:.4f}[{lo:.3f}-{hi:.3f}]  F1={f1:.4f}  "
              f"thr={thr:.2f}  {ds:>8s}  {mk}")
        results[name]={"auc":float(auc),"auc_ci":[float(lo),float(hi)],
                       "f1":float(f1),"accuracy":float(acc),"best_threshold":float(thr),
                       "weights":[float(w1),float(w2),float(w3),float(w4)],
                       "auc_drop":0.0 if base_auc==auc else float(base_auc-auc)}
    return results


def run_metric_ablation(X,y):
    print("\n  [Ablation 2] Individual Metric Contribution (16 features)")
    print("  "+"─"*60)
    bsc=compute_masked_tcs(X,weights=DEFAULT_WEIGHTS)
    bauc=compute_auc(bsc,y); bf1,_=best_f1(bsc,y)
    print(f"    {'Baseline (all 16)':35s}  AUC={bauc:.4f}  F1={bf1:.4f}")
    results={"baseline":{"auc":float(bauc),"f1":float(bf1)}}
    for i,name in enumerate(METRIC_DISPLAY_NAMES):
        mask=np.ones(16,dtype=bool); mask[i]=False
        sc=compute_masked_tcs(X,mask=mask,weights=DEFAULT_WEIGHTS)
        auc=compute_auc(sc,y); f1,_=best_f1(sc,y); drop=bauc-auc
        imp="★★★" if drop>0.05 else "★★" if drop>0.02 else "★" if drop>0.005 else "-"
        print(f"    Remove {name:30s}  AUC={auc:.4f}  drop={drop:+.4f}  F1={f1:.4f}  {imp}")
        results[name]={"auc":float(auc),"f1":float(f1),"drop":float(drop),"importance":imp}
    return results


def run_ctcg_submodule_ablation(X,y):
    print("\n  [Ablation 3] CTCG Sub-Module Contribution (L4 deep-dive)")
    print("  "+"─"*60)
    base_auc=compute_auc(X[:,12],y)
    print(f"    {'CTCG aggregate score':35s}  AUC={base_auc:.4f}")
    subs=[("Phase Coherence",13),("AR Causal Residual",14),("Spectral Graph (Fiedler)",15)]
    results={"ctcg_aggregate":float(base_auc)}
    for desc,idx in subs:
        sc=X[:,idx]; auc=compute_auc(sc,y); f1,thr=best_f1(sc,y)
        lo,hi=bootstrap_auc_ci(sc,y,n=200)
        print(f"    {desc:35s}  AUC={auc:.4f}[{lo:.3f}-{hi:.3f}]  F1={f1:.4f}  thr={thr:.2f}")
        results[desc]={"auc":float(auc),"auc_ci":[float(lo),float(hi)],"f1":float(f1)}
    return results


def run_weight_search(X,y,grid_steps=6):
    print("\n  [Ablation 4] 4-D Weight Grid Search")
    print("  "+"─"*60)
    cands=np.linspace(0,1,grid_steps+1)
    best_auc=-1.0; best_w=None; all_res=[]
    for w1,w2,w3 in product(cands,cands,cands):
        w4=round(1.0-w1-w2-w3,6)
        if w4<0 or w4>1: continue
        sc=compute_tcs(X,float(w1),float(w2),float(w3),float(w4))
        auc=compute_auc(sc,y); f1,thr=best_f1(sc,y)
        all_res.append((float(w1),float(w2),float(w3),float(w4),float(auc),float(f1),float(thr)))
        if auc>best_auc: best_auc=auc; best_w=(float(w1),float(w2),float(w3),float(w4))
    all_res.sort(key=lambda r:(r[4],r[5]),reverse=True)
    print(f"    {'W_L1':>6} {'W_L2':>6} {'W_L3':>6} {'W_L4':>6}  {'AUC':>6}  {'F1':>6}  thr")
    for row in all_res[:10]:
        w1,w2,w3,w4,auc,f1,thr=row
        mk=" ← BEST" if (w1,w2,w3,w4)==best_w else ""
        print(f"    {w1:>6.2f} {w2:>6.2f} {w3:>6.2f} {w4:>6.2f}  {auc:>6.4f}  {f1:>6.4f}  {thr:.2f}{mk}")
    print(f"\n    Optimal: L1={best_w[0]:.2f}  L2={best_w[1]:.2f}  L3={best_w[2]:.2f}  L4={best_w[3]:.2f}")
    print(f"    Default: L1={DEFAULT_WEIGHTS[0]:.2f}  L2={DEFAULT_WEIGHTS[1]:.2f}  "
          f"L3={DEFAULT_WEIGHTS[2]:.2f}  L4={DEFAULT_WEIGHTS[3]:.2f}")
    return {"best_weights":{"L1":best_w[0],"L2":best_w[1],"L3":best_w[2],"L4":best_w[3]},
            "best_auc":float(best_auc),
            "top_results":[{"w1":r[0],"w2":r[1],"w3":r[2],"w4":r[3],
                            "auc":r[4],"f1":r[5],"threshold":r[6]} for r in all_res[:20]]}


def main():
    parser=argparse.ArgumentParser(description="PATV-X+ Ablation Study")
    parser.add_argument("--data"); parser.add_argument("--synthetic",action="store_true")
    parser.add_argument("--output",default="ablation_results.json")
    parser.add_argument("--n-samples",type=int,default=500)
    parser.add_argument("--grid-steps",type=int,default=6)
    parser.add_argument("--split-manifest")
    args=parser.parse_args()
    print("\n"+"="*62)
    print("  PATV-X Core  Ablation Study  (L1+L2+L3+L4 | 16-D)")
    print("="*62)
    if args.synthetic:
        print(f"\n[Data] Synthetic smoke {args.n_samples} samples (16-D)...")
        print("[Note] Synthetic results are smoke-only and not paper-facing.")
        X,y=generate_synthetic_data(args.n_samples); src="synthetic"
    elif args.data:
        print(f"\n[Data] Loading real features: {args.data}")
        X,y=load_csv(args.data); src=str(args.data)
        split_manifest = detect_split_manifest(args.data, args.split_manifest)
        if split_manifest is not None:
            X, y, keep_idx = dev_subset_from_manifest(X, y, split_manifest)
            print(f"[Data] Using development subset from: {split_manifest}  (n={len(keep_idx)})")
    else:
        raise SystemExit("Απαιτείται --data για publishable ablation. Το --synthetic είναι μόνο smoke mode.")
    print(f"[Data] {len(X)} | AI={int(y.sum())} Auth={int((y==0).sum())}")
    X=np.clip(X,0,1)
    r1=run_level_ablation(X,y); r2=run_metric_ablation(X,y)
    r3=run_ctcg_submodule_ablation(X,y); r4=run_weight_search(X,y,args.grid_steps)
    bw=r4["best_weights"]
    print("\n"+"="*62)
    print("  ΣΥΝΟΨΗ")
    print("="*62)
    print(f"\n  Optimal weights: L1={bw['L1']:.2f} L2={bw['L2']:.2f} L3={bw['L3']:.2f} L4={bw['L4']:.2f}")
    print(f"  Best AUC: {r4['best_auc']:.4f}")
    no_ctcg_auc = r1.get("L1+L2+L3 (no CTCG)",{}).get("auc",0)
    all_auc = r1.get("All (L1+L2+L3+L4)",{}).get("auc",0)
    print(f"\n  AUC without CTCG (L1+L2+L3):  {no_ctcg_auc:.4f}")
    print(f"  AUC with    CTCG (L1+L2+L3+L4): {all_auc:.4f}")
    print(f"  CTCG contribution:  {all_auc-no_ctcg_auc:+.4f}")
    drops=[(n,v["drop"]) for n,v in r2.items() if n!="baseline" and isinstance(v,dict) and "drop" in v]
    drops.sort(key=lambda x:x[1],reverse=True)
    print("\n  Top-3 most discriminative features:")
    for nm,dp in drops[:3]: print(f"    {nm}: AUC drop={dp:+.4f}")
    results={"data_source":src,"n_samples":len(X),"n_ai":int(y.sum()),
             "n_authentic":int((y==0).sum()),
             "default_weights":{"L1":DEFAULT_WEIGHTS[0],"L2":DEFAULT_WEIGHTS[1],
                                "L3":DEFAULT_WEIGHTS[2],"L4":DEFAULT_WEIGHTS[3]},
             "feature_names":FEATURE_NAMES,
             "level_ablation":r1,"metric_ablation":r2,
             "ctcg_submodule_ablation":r3,"weight_search":r4}
    with open(args.output,"w",encoding="utf-8") as f:
        json.dump(results,f,indent=2,ensure_ascii=False)
    print(f"\n  Results saved: {args.output}")
    print("="*62+"\n")


if __name__ == "__main__":
    main()
