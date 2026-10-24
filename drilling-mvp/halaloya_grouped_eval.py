#!/usr/bin/env python3
"""HalaLoya grouped-proxy evaluation for recorded severe/complete-loss labels.

RESEARCH ONLY. Inferred depth-reset segments are NOT verified Well IDs.
This is retrospective same-row screening, not future-event early warning.

Example for the Azadegan workbook:
  python halaloya_grouped_eval.py 42452_2021_4769_MOESM1_ESM.xlsx --allow-depth-reset-proxy --expected-critical-count 304 --bootstrap-reps 50 --max-iter 30 --out-json grouped_eval.json --segments-csv inferred_segments.csv

The --allow-depth-reset-proxy flag is deliberately required. A segment count close to
20 is not enough to claim that the segments are the 20 physical wells.
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

TARGET_NAME = "lossesseverity"
DEPTH_NAME = "mdepth"
MUD_HINTS = ("mud", "funnel", "plastic", "yield", "gel", "solid")
SEVERE_WORDS = {"severe loss", "complete loss", "severe", "complete"}

def norm_name(value: object) -> str:
    return "".join(ch.lower() for ch in str(value) if ch.isalnum())

def load_data(path: str, sheet: str | None) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"Input file not found: {p}")
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    if p.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
        return pd.read_excel(p, sheet_name=sheet if sheet is not None else 0)
    raise SystemExit("Unsupported file type. Use CSV or Excel (.xlsx/.xlsm/.xls).")

def find_column(df: pd.DataFrame, desired_normalized: str) -> str:
    matches = [c for c in df.columns if norm_name(c) == desired_normalized]
    if not matches:
        raise SystemExit(f"Required column matching {desired_normalized!r} not found. Available columns: {list(df.columns)}")
    if len(matches) > 1:
        raise SystemExit(f"Multiple columns match {desired_normalized!r}: {matches}")
    return matches[0]

def severe_flag(series: pd.Series) -> pd.Series:
    """Recognize text labels, whitespace/underscore variants and numeric codes 4/5."""
    def is_critical(value: object) -> bool:
        if pd.isna(value):
            return False
        label = re.sub(r"[\s_\-]+", " ", str(value).strip().lower()).strip()
        if label in SEVERE_WORDS or "severe loss" in label or "complete loss" in label:
            return True
        try:
            numeric = float(str(value).strip())
            return bool(np.isfinite(numeric) and numeric in (4.0, 5.0))
        except (TypeError, ValueError):
            return False
    return series.map(is_critical).astype(int)

def build_segments(depth: np.ndarray, drop_m: float) -> np.ndarray:
    return np.concatenate([[0], np.cumsum(np.diff(depth) < -drop_m)]).astype(int)

def segment_summaries(df: pd.DataFrame, target_col: str, depth: np.ndarray, groups: np.ndarray) -> list[dict[str, Any]]:
    critical = severe_flag(df[target_col]).to_numpy(dtype=int)
    complete = df[target_col].astype(str).str.strip().str.lower().eq("complete loss").to_numpy()
    out = []
    for group in np.unique(groups):
        idx = np.flatnonzero(groups == group)
        out.append({
            "proxy_segment": int(group), "first_row_1based": int(idx[0] + 1),
            "last_row_1based": int(idx[-1] + 1), "rows": int(len(idx)),
            "start_depth_m": float(depth[idx[0]]), "end_depth_m": float(depth[idx[-1]]),
            "critical_rows": int(critical[idx].sum()), "complete_rows": int(complete[idx].sum()),
        })
    return out

def boundary_diagnostics(df: pd.DataFrame, depth_col: str, drop_m: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    depth = pd.to_numeric(df[depth_col], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(depth).all():
        raise SystemExit("Missing/non-numeric measured depths exist; fix before attempting row-order groups.")
    d_diff = np.diff(depth)
    hole_col = next((c for c in df.columns if norm_name(c) == "holesection"), None)
    mud_cols = [c for c in df.columns if any(h in str(c).lower() for h in MUD_HINTS) and c != depth_col]
    boundaries = []
    for after_idx in np.flatnonzero(d_diff < -drop_m) + 1:
        hole_changed = bool(df[hole_col].iloc[after_idx - 1] != df[hole_col].iloc[after_idx]) if hole_col is not None else None
        mud_changed = bool(any(df[col].iloc[after_idx - 1] != df[col].iloc[after_idx] for col in mud_cols)) if mud_cols else None
        boundaries.append({
            "row_before_1based": int(after_idx), "row_after_1based": int(after_idx + 1),
            "depth_before_m": float(depth[after_idx - 1]), "depth_after_m": float(depth[after_idx]),
            "depth_drop_m": float(depth[after_idx] - depth[after_idx - 1]),
            "hole_section_changed": hole_changed, "any_mud_property_changed": mud_changed,
        })
    segment_ids = build_segments(depth, drop_m)
    diag = {
        "rows": int(len(depth)),
        "share_non_decreasing_steps": float((d_diff >= 0).mean()),
        "share_abs_depth_steps_le_5m": float((np.abs(d_diff) <= 5).mean()),
        "number_of_drops_over_threshold": int((d_diff < -drop_m).sum()),
        "segment_count": int(len(np.unique(segment_ids))),
        "segments_starting_deeper_than_500m": int(sum(depth[np.flatnonzero(segment_ids == group)[0]] > 500 for group in np.unique(segment_ids))),
        "boundaries_with_hole_section_change": int(sum(b["hole_section_changed"] is True for b in boundaries)),
        "boundaries_with_any_mud_property_change": int(sum(b["any_mud_property_changed"] is True for b in boundaries)),
        "threshold_sensitivity": [],
        "shuffled_order_negative_control": {},
    }
    target_col = find_column(df, TARGET_NAME)
    y = severe_flag(df[target_col]).to_numpy()
    complete = df[target_col].astype(str).str.strip().str.lower().eq("complete loss").to_numpy()
    for threshold in (10.0, 50.0, 200.0, 500.0):
        seg = build_segments(depth, threshold)
        diag["threshold_sensitivity"].append({
            "drop_threshold_m": threshold,
            "segment_count": int(len(np.unique(seg))),
            "segments_with_critical_rows": int(pd.Series(seg[y == 1]).nunique()),
            "segments_with_complete_rows": int(pd.Series(seg[complete]).nunique()),
        })
    rng = np.random.default_rng(2026)
    shuffled = rng.permutation(depth)
    shuffled_diff = np.diff(shuffled)
    shuffled_groups = build_segments(shuffled, 50.0)
    diag["shuffled_order_negative_control"] = {
        "seed": 2026,
        "share_non_decreasing_steps": float((shuffled_diff >= 0).mean()),
        "share_abs_depth_steps_le_5m": float((np.abs(shuffled_diff) <= 5).mean()),
        "drops_over_50m": int((shuffled_diff < -50).sum()),
        "segment_count_at_50m": int(len(np.unique(shuffled_groups))),
        "interpretation": "A sharp difference supports that the original is depth-ordered; it does not prove each inferred segment is a real well.",
    }
    return boundaries, diag

def nearest_neighbor_overlap_audit(df: pd.DataFrame, groups: np.ndarray, sample_size: int = 1000, seed: int = 0) -> dict[str, Any]:
    """Exploratory similarity diagnostic; not proof of well identity or leakage."""
    cols = [c for c in df.select_dtypes(include="number").columns if norm_name(c) not in {TARGET_NAME, DEPTH_NAME}]
    if len(cols) < 2:
        return {"available": False, "reason": "Too few numeric features."}
    frame = df[cols].replace([np.inf, -np.inf], np.nan)
    frame = frame.fillna(frame.median(numeric_only=True)).fillna(0)
    n = min(sample_size, len(frame))
    rng = np.random.default_rng(seed)
    sample_idx = np.sort(rng.choice(len(frame), size=n, replace=False))
    x = RobustScaler().fit_transform(frame.iloc[sample_idx].to_numpy(dtype=float))
    g = groups[sample_idx]
    if len(np.unique(g)) < 2 or n < 20:
        return {"available": False, "reason": "Too few sample rows or segments."}
    nn = NearestNeighbors(n_neighbors=min(12, n), algorithm="brute", n_jobs=1)
    nn.fit(x)
    distances, neighbors = nn.kneighbors(x)
    same, cross, paired = [], [], []
    for i in range(n):
        same_dist = next((float(dist) for dist, j in zip(distances[i, 1:], neighbors[i, 1:]) if g[i] == g[j]), None)
        cross_dist = next((float(dist) for dist, j in zip(distances[i, 1:], neighbors[i, 1:]) if g[i] != g[j]), None)
        if same_dist is not None:
            same.append(same_dist)
        if cross_dist is not None:
            cross.append(cross_dist)
        if same_dist is not None and cross_dist is not None:
            paired.append(cross_dist <= same_dist)
    return {
        "available": bool(same and cross), "sample_rows": int(n), "features_used": [str(c) for c in cols],
        "same_proxy_segment_neighbor_rows": int(len(same)), "cross_proxy_segment_neighbor_rows": int(len(cross)),
        "paired_rows_compared": int(len(paired)),
        "same_proxy_segment_nearest_neighbor_distance_median": float(np.median(same)) if same else None,
        "cross_proxy_segment_nearest_neighbor_distance_median": float(np.median(cross)) if cross else None,
        "fraction_paired_rows_with_cross_segment_distance_le_same_segment_distance": float(np.mean(paired)) if paired else None,
        "interpretation": "Close cross-segment neighbors flag feature-space overlap; they do not establish the same physical well or prove leakage.",
    }

def segment_adversarial_auc(df: pd.DataFrame, target_col: str, groups: np.ndarray, max_rows_per_side: int = 1500, seed: int = 123) -> dict[str, Any]:
    """One-vs-rest segment-membership separability; NOT a test of well identity."""
    cols = [c for c in df.select_dtypes(include="number").columns if norm_name(c) not in {TARGET_NAME, DEPTH_NAME}]
    x = df[cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    records = []
    for group in np.unique(groups):
        positive = np.flatnonzero(groups == group)
        negative = np.flatnonzero(groups != group)
        n = min(max_rows_per_side, len(positive), len(negative))
        if n < 30:
            records.append({"proxy_segment": int(group), "rows_sampled_per_side": int(n), "roc_auc": None})
            continue
        idx = np.concatenate([rng.choice(positive, n, replace=False), rng.choice(negative, n, replace=False)])
        labels = np.concatenate([np.ones(n, dtype=int), np.zeros(n, dtype=int)])
        train_idx, test_idx = train_test_split(np.arange(len(labels)), test_size=0.30, random_state=42, stratify=labels)
        model = make_pipeline(SimpleImputer(strategy="median"), RobustScaler(), LogisticRegression(max_iter=300, class_weight="balanced", C=1.0))
        model.fit(x[idx][train_idx], labels[train_idx])
        auc = float(roc_auc_score(labels[test_idx], model.predict_proba(x[idx][test_idx])[:, 1]))
        records.append({"proxy_segment": int(group), "rows_sampled_per_side": int(n), "roc_auc": auc})
    aucs = [row["roc_auc"] for row in records if row["roc_auc"] is not None]
    return {
        "purpose": "adversarial distribution-separability check: each proxy segment versus all other segments",
        "features_exclude_target_and_depth": [str(c) for c in cols],
        "segment_results": records,
        "median_auc": float(np.median(aucs)) if aucs else None,
        "min_auc": float(np.min(aucs)) if aucs else None,
        "max_auc": float(np.max(aucs)) if aucs else None,
        "interpretation": "High AUC means the segment's feature distribution is distinguishable from other segments. AUC near 0.5 means not distinguishable in this setup. Neither result proves physical well identity.",
    }

def numeric_feature_columns(df: pd.DataFrame, target_col: str, variant: str) -> list[str]:
    excluded = {TARGET_NAME, DEPTH_NAME}
    if variant in {"without_flow_family", "without_flow_family_and_spp"}:
        excluded.update({"flowin", "flowout", "pumpstroke"})
    if variant == "without_flow_family_and_spp":
        excluded.add("standpipepressure")
    return [c for c in df.select_dtypes(include="number").columns if c != target_col and norm_name(c) not in excluded]

def loo_probabilities(df: pd.DataFrame, groups: np.ndarray, y: np.ndarray, columns: list[str], max_iter: int) -> tuple[np.ndarray, list[dict[str, Any]], list[int]]:
    x = df[columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    scores = np.full(len(y), np.nan, dtype=float)
    skipped, iterations = [], []
    for group in np.unique(groups):
        test = groups == group
        train = ~test
        if y[train].sum() < 5 or np.unique(y[train]).size < 2:
            skipped.append({"proxy_segment": int(group), "training_positives": int(y[train].sum()), "reason": "too few positives or one class in training"})
            continue
        model = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=max_iter, class_weight="balanced", random_state=0, early_stopping=False)
        model.fit(x[train], y[train])
        pos_idx = list(model.classes_).index(1)
        scores[test] = model.predict_proba(x[test])[:, pos_idx]
        iterations.append(int(model.n_iter_))
    return scores, skipped, iterations

def contiguous_positive_runs(y_group: np.ndarray) -> list[tuple[int, int]]:
    starts = np.flatnonzero(y_group.astype(bool) & ~np.r_[False, y_group[:-1].astype(bool)])
    runs = []
    for start in starts:
        end = int(start)
        while end + 1 < len(y_group) and y_group[end + 1] == 1:
            end += 1
        runs.append((int(start), end))
    return runs

def per_group_metrics(y: np.ndarray, scores: np.ndarray, groups: np.ndarray) -> dict[int, dict[str, Any]]:
    stats = {}
    for group in np.unique(groups):
        idx = np.flatnonzero((groups == group) & np.isfinite(scores))
        if not len(idx):
            continue
        yg, sg = y[idx], scores[idx]
        positives = int(yg.sum())
        ap = float(average_precision_score(yg, sg)) if positives else None
        runs = contiguous_positive_runs(yg)
        item = {
            "proxy_segment": int(group), "rows": int(len(idx)), "critical_rows": positives,
            "prevalence": float(yg.mean()), "average_precision_within_segment": ap,
            "ap_minus_segment_prevalence": float(ap - yg.mean()) if ap is not None else None,
            "proxy_positive_runs": int(len(runs)), "budgets": {},
        }
        for budget in (0.01, 0.02, 0.05):
            k = max(1, int(np.ceil(len(idx) * budget)))
            selected = np.argsort(-sg, kind="stable")[:k]
            alerted = np.zeros(len(idx), dtype=bool)
            alerted[selected] = True
            tp = int(yg[selected].sum())
            detected_runs = sum(bool(alerted[a:b + 1].any()) for a, b in runs)
            key = f"top_{int(budget * 100)}pct"
            item["budgets"][key] = {
                "budget": budget, "alerts": int(k), "true_positives": tp,
                "false_positives": int(k - tp),
                "row_recall": float(tp / positives) if positives else None,
                "alert_precision": float(tp / k) if k else None,
                "proxy_runs_detected": int(detected_runs),
                "proxy_run_recall": float(detected_runs / len(runs)) if runs else None,
            }
        stats[int(group)] = item
    return stats

def bootstrap_group_stats(stats: dict[int, dict[str, Any]], n: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    group_ids = np.array(list(stats.keys()), dtype=int)
    metric_names = ["macro_ap_minus_local_prevalence", "row_recall_top_1pct", "row_recall_top_2pct", "row_recall_top_5pct",
                    "precision_top_1pct", "precision_top_2pct", "precision_top_5pct",
                    "proxy_run_recall_top_1pct", "proxy_run_recall_top_2pct", "proxy_run_recall_top_5pct"]
    values = {name: [] for name in metric_names}
    zero_positive_reps = 0
    for _ in range(n):
        picked = rng.choice(group_ids, size=len(group_ids), replace=True)
        selected_stats = [stats[int(g)] for g in picked]
        positive_stats = [item for item in selected_stats if item["critical_rows"] > 0 and item["ap_minus_segment_prevalence"] is not None]
        if not positive_stats:
            zero_positive_reps += 1
        else:
            values["macro_ap_minus_local_prevalence"].append(float(np.mean([x["ap_minus_segment_prevalence"] for x in positive_stats])))
        positives = sum(item["critical_rows"] for item in selected_stats)
        for pct in (1, 2, 5):
            key = f"top_{pct}pct"
            tp = sum(item["budgets"][key]["true_positives"] for item in selected_stats)
            alerts = sum(item["budgets"][key]["alerts"] for item in selected_stats)
            events_detected = sum(item["budgets"][key]["proxy_runs_detected"] for item in selected_stats)
            event_total = sum(item["proxy_positive_runs"] for item in selected_stats)
            values[f"row_recall_{key}"].append(float(tp / positives) if positives else float("nan"))
            values[f"precision_{key}"].append(float(tp / alerts) if alerts else float("nan"))
            values[f"proxy_run_recall_{key}"].append(float(events_detected / event_total) if event_total else float("nan"))
    def summarize(array: list[float]) -> dict[str, Any]:
        vals = np.asarray(array, dtype=float)
        vals = vals[np.isfinite(vals)]
        if not len(vals):
            return {"ci_2.5": None, "median": None, "ci_97.5": None, "valid_replicates": 0}
        lo, median, hi = np.percentile(vals, [2.5, 50, 97.5])
        return {"ci_2.5": float(lo), "median": float(median), "ci_97.5": float(hi), "valid_replicates": int(len(vals))}
    return {
        "requested_replicates": int(n),
        "replicates_with_no_positive_groups_excluded_from_macro_ap_delta": int(zero_positive_reps),
        "metrics": {name: summarize(array) for name, array in values.items()},
        "method_note": "Bootstrap resamples inferred proxy segments and holds out-of-segment scores fixed. Macro AP delta averages segment AP minus its local prevalence over sampled groups with positive labels. Replicates with no positive groups are counted and excluded from this undefined metric; intervals do not include full model-fitting uncertainty.",
    }

def evaluate_variant(name: str, y: np.ndarray, scores: np.ndarray, groups: np.ndarray, bootstrap_reps: int, seed: int) -> dict[str, Any]:
    valid = np.isfinite(scores)
    yv, sv, gv = y[valid], scores[valid], groups[valid]
    prevalence = float(yv.mean())
    stats = per_group_metrics(yv, sv, gv)
    positive_stats = [item for item in stats.values() if item["critical_rows"] > 0]
    pooled_ap = float(average_precision_score(yv, sv)) if yv.sum() else None
    boot = bootstrap_group_stats(stats, bootstrap_reps, seed)
    return {
        "name": name, "rows_scored": int(len(yv)), "critical_rows_scored": int(yv.sum()), "prevalence_scored": prevalence,
        "always_noncritical_accuracy": float(1.0 - prevalence), "always_noncritical_critical_recall": 0.0,
        "pooled_oof_average_precision_secondary_uncalibrated_across_fold_models": pooled_ap,
        "pooled_ap_minus_global_prevalence_secondary": float(pooled_ap - prevalence) if pooled_ap is not None else None,
        "positive_proxy_segments_scored": int(len(positive_stats)),
        "macro_ap_within_positive_proxy_segments": float(np.mean([item["average_precision_within_segment"] for item in positive_stats])) if positive_stats else None,
        "macro_ap_minus_local_prevalence_within_positive_segments": float(np.mean([item["ap_minus_segment_prevalence"] for item in positive_stats])) if positive_stats else None,
        "bootstrap": boot, "per_proxy_segment": list(stats.values()),
        "limitations": [
            "Pooled OOF scores come from different fold-specific models and are not calibrated across folds; pooled AP is secondary.",
            "Primary ranking summaries are computed within each held-out proxy segment, then aggregated.",
            "Positive runs are contiguous positive-label rows in row order; they are proxy episodes, not verified physical loss events or timed events.",
        ],
    }

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument("--sheet", default=None)
    parser.add_argument("--drop", type=float, default=50.0, help="Depth-reset threshold in meters")
    parser.add_argument("--expected-critical-count", type=int, default=None, help="Safety check; use 304 for the supplied Azadegan workbook")
    parser.add_argument("--expect-wells", type=int, default=20, help="Reference count for context only; never certifies group identity")
    parser.add_argument("--allow-depth-reset-proxy", action="store_true", help="Explicitly allow inferred depth-reset proxy groups; these are NOT real Well IDs")
    parser.add_argument("--allow-mud-signature-proxy", action="store_true", help="Use rounded mud signatures only as an explicitly labelled proxy, never as well/day IDs")
    parser.add_argument("--max-proxy-groups", type=int, default=100)
    parser.add_argument("--bootstrap-reps", type=int, default=50)
    parser.add_argument("--max-iter", type=int, default=30, help="Fixed boosting iterations per held-out group; early stopping is disabled")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-neighbors", type=int, default=1000)
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--segments-csv", default=None)
    args = parser.parse_args()

    raw = load_data(args.path, args.sheet)
    target_col, depth_col = find_column(raw, TARGET_NAME), find_column(raw, DEPTH_NAME)
    if raw[target_col].isna().any():
        raise SystemExit(f"Target contains {int(raw[target_col].isna().sum())} missing labels; review before evaluation.")
    if pd.to_numeric(raw[depth_col], errors="coerce").isna().any():
        raise SystemExit("Missing or non-numeric depth values exist; review before inferred segmentation.")
    df = raw.copy().reset_index(drop=True)
    df[depth_col] = pd.to_numeric(df[depth_col], errors="coerce").astype(float)
    y = severe_flag(df[target_col]).to_numpy(dtype=int)
    if int(y.sum()) == 0:
        raise SystemExit("No Severe/Complete labels were recognized. Observed values: " + repr(df[target_col].astype(str).value_counts().to_dict()))
    if args.expected_critical_count is not None and int(y.sum()) != args.expected_critical_count:
        raise SystemExit(f"Label guard failed: recognized {int(y.sum())} critical rows but expected {args.expected_critical_count}. Check target values/encoding; no model was fitted.")

    depth = df[depth_col].to_numpy(dtype=float)
    boundary_rows, diag = boundary_diagnostics(df, depth_col, args.drop)
    if args.allow_depth_reset_proxy:
        groups = build_segments(depth, args.drop)
        grouping_method = f"depth-reset segments at {args.drop:g}m (proxy only; not confirmed well IDs)"
    elif args.allow_mud_signature_proxy:
        mud_cols = [c for c in df.columns if any(h in str(c).lower() for h in MUD_HINTS) and c != target_col and norm_name(c) != DEPTH_NAME]
        if not mud_cols:
            raise SystemExit("No mud-property columns found for fallback proxy.")
        signature = df[mud_cols].apply(pd.to_numeric, errors="coerce").round(2).astype("string").fillna("<NA>").agg("|".join, axis=1)
        groups, _ = pd.factorize(signature, sort=False)
        if len(np.unique(groups)) > args.max_proxy_groups:
            raise SystemExit(f"Mud-signature proxy produced {len(np.unique(groups)):,} groups, above {args.max_proxy_groups}. This is not a substitute for Well IDs.")
        grouping_method = f"mud-signature proxy {mud_cols}; NOT wells or days"
    else:
        raise SystemExit("No grouping selected. Use --allow-depth-reset-proxy for clearly labelled proxy analysis, or provide real Well IDs. A segment count near 20 does not certify well identity.")

    class_counts = df[target_col].astype(str).value_counts().to_dict()
    critical = int(y.sum())
    positive_groups = int(pd.Series(groups[y == 1]).nunique())
    complete_mask = df[target_col].astype(str).str.strip().str.lower().eq("complete loss").to_numpy()
    complete_groups = int(pd.Series(groups[complete_mask]).nunique())
    segments = segment_summaries(df, target_col, depth, groups)
    sizes = pd.Series(groups).value_counts()
    neighbor_audit = nearest_neighbor_overlap_audit(df, groups, args.sample_neighbors, args.seed)
    adversarial_audit = segment_adversarial_auc(df, target_col, groups, seed=args.seed + 123)

    print(f"rows={len(df):,}; columns={len(df.columns)}; missing_cells={int(df.isna().sum().sum()):,}; exact_duplicates={int(df.duplicated().sum()):,}")
    print("class counts:", json.dumps({str(k): int(v) for k, v in class_counts.items()}, ensure_ascii=False))
    print(f"critical labels recognized={critical} ({critical/len(df):.3%}); always-noncritical accuracy={1-critical/len(df):.3%}; always-noncritical recall=0")
    print("order diagnostics:", json.dumps(diag, indent=2, ensure_ascii=False))
    print(f"groups={len(np.unique(groups))} [{grouping_method}]; groups with critical rows={positive_groups}; groups with complete-loss rows={complete_groups}")
    if positive_groups <= 5:
        print("CAUTION: very few proxy groups contain positive labels. Group-level uncertainty will be wide and group identity is unverified.")
    if diag["shuffled_order_negative_control"]["segment_count_at_50m"] > 1000:
        print("Shuffled-order negative control differs strongly from original; this supports depth-ordered rows, not true well identity.")
    print("nearest-neighbor overlap audit:", json.dumps(neighbor_audit, indent=2))
    print("adversarial segment-distribution audit:", json.dumps({k: v for k, v in adversarial_audit.items() if k != "segment_results"}, indent=2))
    print("Flow-unit caution: published table describes FlowIn in gpm and FlowOut in percent; never subtract them directly.")

    results = []
    variants = [
        ("all_numeric_except_target_and_depth", "all"),
        ("without_FlowIn_FlowOut_PumpStroke_family", "without_flow_family"),
        ("without_flow_family_and_StandpipePressure", "without_flow_family_and_spp"),
    ]
    for offset, (name, variant) in enumerate(variants):
        cols = numeric_feature_columns(df, target_col, variant)
        scores, skipped, iterations = loo_probabilities(df, groups, y, cols, args.max_iter)
        if skipped:
            print(f"WARNING {name}: skipped {len(skipped)} groups: {skipped}")
        valid = np.isfinite(scores)
        if not valid.any() or y[valid].sum() == 0:
            print(f"Skipping {name}: no scorable critical rows.")
            continue
        result = evaluate_variant(name, y[valid], scores[valid], groups[valid], args.bootstrap_reps, args.seed + offset)
        result.update({
            "features": [str(c) for c in cols], "grouping_method": grouping_method,
            "depth_excluded_from_features": True, "max_iter_fixed_per_fold": int(args.max_iter),
            "early_stopping": False,
            "iterations_per_fold": {"min": int(min(iterations)) if iterations else None, "median": float(np.median(iterations)) if iterations else None, "max": int(max(iterations)) if iterations else None},
            "skipped_folds": skipped,
        })
        print(f"\n{name}: scored rows={result['rows_scored']:,}, positives={result['critical_rows_scored']}, prevalence={result['prevalence_scored']:.4f}")
        print(f"  pooled OOF AP (secondary/uncalibrated)={result['pooled_oof_average_precision_secondary_uncalibrated_across_fold_models']:.4f}; pooled AP minus prevalence={result['pooled_ap_minus_global_prevalence_secondary']:.4f}")
        print(f"  macro AP within positive proxy segments={result['macro_ap_within_positive_proxy_segments']:.4f}; mean per-segment AP minus local prevalence={result['macro_ap_minus_local_prevalence_within_positive_segments']:.4f}")
        for key, metric in result["bootstrap"]["metrics"].items():
            print(f"  {key}: median={metric['median']} 95% CI=({metric['ci_2.5']}, {metric['ci_97.5']}) valid bootstrap reps={metric['valid_replicates']}")
        print(f"  bootstrap requested={result['bootstrap']['requested_replicates']}; no-positive-group reps excluded from macro AP delta={result['bootstrap']['replicates_with_no_positive_groups_excluded_from_macro_ap_delta']}")
        results.append(result)

    flowout_col = next((c for c in df.columns if norm_name(c) == "flowout"), None)
    if flowout_col is not None:
        flowout = pd.to_numeric(df[flowout_col], errors="coerce")
        in_range = flowout.notna() & flowout.between(0, 100)
        heuristic_scores = np.full(len(df), np.nan)
        heuristic_scores[in_range.to_numpy()] = 100.0 - flowout[in_range].to_numpy(dtype=float)
        valid = np.isfinite(heuristic_scores)
        if valid.any() and y[valid].sum() > 0:
            heuristic = evaluate_variant("FlowOut-only ranking heuristic; not an alarm rule", y[valid], heuristic_scores[valid], groups[valid], args.bootstrap_reps, args.seed + 991)
            heuristic.update({"features": [str(flowout_col)], "interpretation": "Directional benchmark only; verify sensor semantics and operating context with a drilling engineer."})
            print(f"\nFlowOut-only heuristic: pooled AP secondary={heuristic['pooled_oof_average_precision_secondary_uncalibrated_across_fold_models']:.4f}; proxy-run recall@5% within each segment={heuristic['bootstrap']['metrics']['proxy_run_recall_top_5pct']['median']}")
            results.append(heuristic)

    report = {
        "project": "HalaLoya AI Drilling Assistant",
        "task": "Retrospective same-row Severe/Complete Loss screening; not early warning",
        "input_file": Path(args.path).name, "rows": int(len(df)), "columns": int(len(df.columns)),
        "missing_cells": int(df.isna().sum().sum()), "exact_duplicates": int(df.duplicated().sum()),
        "class_counts": {str(k): int(v) for k, v in class_counts.items()},
        "critical_rows": critical, "critical_prevalence": float(y.mean()),
        "always_noncritical_accuracy": float(1-y.mean()),
        "grouping_method": grouping_method, "group_count": int(pd.Series(groups).nunique()),
        "groups_with_critical_rows": positive_groups, "groups_with_complete_loss": complete_groups,
        "fixed_max_iter": int(args.max_iter), "early_stopping": False,
        "segment_summaries": segments, "boundary_diagnostics": boundary_rows,
        "order_diagnostics": diag, "nearest_neighbor_overlap_audit": neighbor_audit,
        "adversarial_segment_distribution_audit": adversarial_audit, "results": results,
        "limitations": [
            "Inferred depth-reset segments are proxy groups, not confirmed Well IDs; count similar to 20 does not certify identities.",
            "Pooled OOF probabilities come from different fold-specific models; pooled AP is secondary and may be distorted by score-scale differences.",
            "Primary ranking summaries are within each held-out proxy segment.",
            "Contiguous positive-label runs are proxy episodes, not verified physical loss events or timed events.",
            "Only a few proxy groups contain positives; group-bootstrap uncertainty is wide and conditional on fixed OOF scores.",
            "Adversarial AUC and nearest-neighbor similarity diagnose distribution shift/overlap but cannot prove well identity.",
            "FlowIn is reported in gpm and FlowOut in percent; direct subtraction is invalid.",
        ],
    }
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        print(f"\nJSON report written to {args.out_json}")
    if args.segments_csv:
        pd.DataFrame(segments).to_csv(args.segments_csv, index=False)
        print(f"Segment table written to {args.segments_csv}")

if __name__ == "__main__":
    main()
