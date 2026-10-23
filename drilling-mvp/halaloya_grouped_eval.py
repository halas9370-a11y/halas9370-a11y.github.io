#!/usr/bin/env python3
"""HalaLoya: grouped/blocked evaluation for severe + complete loss classification.

RESEARCH ONLY. A row-order segment is only a proxy until mapped to real Well IDs.
The mud-signature fallback is explicitly opt-in and is NOT a well/day identifier.

Usage:
  python halaloya_grouped_eval.py data.xlsx
  python halaloya_grouped_eval.py data.xlsx --sheet Sheet1
  python halaloya_grouped_eval.py data.xlsx --bootstrap-reps 300

Outputs grouped leave-one-group-out scores for (a) numeric features excluding target
and M.Depth, and (b) the same features excluding FlowIn/FlowOut. Reports average
precision relative to prevalence and recall within the top 1%, 2%, and 5% alert budget,
with bootstrap resampling by group. This evaluates retrospective same-row detection,
not future-event early warning.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

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
        raise SystemExit(
            f"Required column matching {desired_normalized!r} not found. "
            f"Available columns: {list(df.columns)}"
        )
    if len(matches) > 1:
        raise SystemExit(f"Multiple columns match {desired_normalized!r}: {matches}")
    return matches[0]


def severe_flag(series: pd.Series) -> pd.Series:
    """Handle text labels, integers, floats (4.0/5.0), and numeric strings."""
    def is_critical(v: object) -> bool:
        if pd.isna(v):
            return False
        label = re.sub(r"[\s_\-]+", " ", str(v).strip().lower()).strip()
        if label in SEVERE_WORDS or "severe loss" in label or "complete loss" in label:
            return True
        try:
            number = float(str(v).strip())
            return np.isfinite(number) and number in (4.0, 5.0)
        except (TypeError, ValueError):
            return False
    return series.map(is_critical).astype(int)


def order_diagnostics(depth: pd.Series, drop_m: float) -> tuple[dict, np.ndarray]:
    d = pd.to_numeric(depth, errors="coerce").to_numpy(dtype=float)
    if len(d) < 3 or not np.isfinite(d).all():
        raise SystemExit("Missing/non-numeric depth values must be reviewed before segment reconstruction.")
    diff = np.diff(d)
    seg = np.concatenate([[0], np.cumsum(diff < -drop_m)])
    sizes = pd.Series(seg).value_counts()
    diag = {
        "rows_with_valid_depth": int(len(d)),
        "share_non_decreasing_steps": float((diff >= 0).mean()),
        "share_abs_depth_steps_le_5m": float((np.abs(diff) <= 5).mean()),
        "large_depth_resets_below_minus_drop_m": int((diff < -drop_m).sum()),
        "segment_count_from_depth_resets": int(len(np.unique(seg))),
        "segment_size_min": int(sizes.min()),
        "segment_size_median": float(sizes.median()),
        "segment_size_max": int(sizes.max()),
    }
    return diag, seg.astype(int)


def mud_signature_groups(df: pd.DataFrame, target_col: str) -> tuple[np.ndarray, list[str]]:
    cols = [c for c in df.columns if any(h in str(c).lower() for h in MUD_HINTS) and c != target_col]
    cols = [c for c in cols if norm_name(c) != DEPTH_NAME]
    if not cols:
        raise SystemExit(f"No mud-property columns found for an opt-in proxy. Columns: {list(df.columns)}")
    signature = df[cols].apply(pd.to_numeric, errors="coerce").round(2).astype("string").fillna("<NA>")
    signature_key = signature.agg("|".join, axis=1)
    groups, _ = pd.factorize(signature_key, sort=False)
    return groups.astype(int), cols


def numeric_feature_columns(df: pd.DataFrame, target_col: str, without_flow: bool) -> list[str]:
    cols = []
    for col in df.select_dtypes(include="number").columns:
        normalized = norm_name(col)
        if col == target_col or normalized in {TARGET_NAME, DEPTH_NAME}:
            continue
        if without_flow and normalized in {"flowin", "flowout"}:
            continue
        cols.append(col)
    return cols


def loo_probabilities(df: pd.DataFrame, groups: np.ndarray, y: np.ndarray,
                      columns: list[str]) -> tuple[np.ndarray, list[str]]:
    Xdf = df[columns].apply(pd.to_numeric, errors="coerce")
    Xdf = Xdf.replace([np.inf, -np.inf], np.nan)
    X = Xdf.to_numpy(dtype=float)
    scores = np.full(len(y), np.nan, dtype=float)
    skipped: list[str] = []
    for group in np.unique(groups):
        test_mask = groups == group
        train_mask = ~test_mask
        if y[train_mask].sum() < 5 or np.unique(y[train_mask]).size < 2:
            skipped.append(str(group))
            continue
        model = HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.05, max_iter=200,
            class_weight="balanced", random_state=0,
        )
        model.fit(X[train_mask], y[train_mask])
        pos_index = list(model.classes_).index(1)
        scores[test_mask] = model.predict_proba(X[test_mask])[:, pos_index]
    return scores, skipped


def group_bootstrap_all(y: np.ndarray, score: np.ndarray, groups: np.ndarray,
                        n: int = 50, seed: int = 0) -> dict:
    """Bootstrap groups once per replicate and calculate all ranking metrics together."""
    rng = np.random.default_rng(seed)
    unique_groups = pd.unique(groups)
    row_indices = {g: np.flatnonzero(groups == g) for g in unique_groups}
    values = {"average_precision": [], "recall_top_1pct": [], "recall_top_2pct": [], "recall_top_5pct": []}
    for _ in range(n):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([row_indices[g] for g in sampled_groups])
        yb, sb = y[indices], score[indices]
        positive_total = int(yb.sum())
        if positive_total == 0:
            continue
        values["average_precision"].append(float(average_precision_score(yb, sb)))
        order = np.argsort(-sb, kind="stable")
        for key, budget in (("recall_top_1pct", 0.01), ("recall_top_2pct", 0.02), ("recall_top_5pct", 0.05)):
            k = max(1, int(np.ceil(len(yb) * budget)))
            values[key].append(float(yb[order[:k]].sum() / positive_total))
    out = {}
    for key, array in values.items():
        if not array:
            out[key] = {"ci_2.5": float("nan"), "median": float("nan"), "ci_97.5": float("nan")}
        else:
            lo, median, hi = np.percentile(array, [2.5, 50, 97.5])
            out[key] = {"ci_2.5": float(lo), "median": float(median), "ci_97.5": float(hi)}
    return out


def evaluate_score(name: str, y: np.ndarray, score: np.ndarray, groups: np.ndarray,
                   bootstrap_reps: int, seed: int) -> dict:
    valid = np.isfinite(score)
    y, score, groups = y[valid], score[valid], groups[valid]
    prevalence = float(y.mean())
    boot = group_bootstrap_all(y, score, groups, bootstrap_reps, seed)
    ap_median = boot["average_precision"]["median"]
    return {
        "name": name,
        "rows_scored": int(len(y)),
        "positive_rows": int(y.sum()),
        "prevalence": prevalence,
        "average_precision_full_oof": float(average_precision_score(y, score)),
        "average_precision_group_bootstrap_ci_2.5": boot["average_precision"]["ci_2.5"],
        "average_precision_group_bootstrap_median": ap_median,
        "average_precision_group_bootstrap_ci_97.5": boot["average_precision"]["ci_97.5"],
        "average_precision_lift_over_prevalence": float(ap_median / prevalence) if prevalence > 0 else None,
        "recall_at_alert_budgets": {
            "recall_top_1pct": {**boot["recall_top_1pct"], "budget": 0.01},
            "recall_top_2pct": {**boot["recall_top_2pct"], "budget": 0.02},
            "recall_top_5pct": {**boot["recall_top_5pct"], "budget": 0.05},
        },
        "interpretation_note": "Group bootstrap is conditional on generated out-of-group scores; it does not fully capture model-training uncertainty.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument("--sheet", default=None)
    parser.add_argument("--drop", type=float, default=50.0, help="Depth fall threshold (meters) used to hypothesize segment resets")
    parser.add_argument("--expect-wells", type=int, default=20)
    parser.add_argument("--allow-mud-signature-proxy", action="store_true", help="Opt into grouping by rounded mud signatures; these are NOT well IDs or days")
    parser.add_argument("--max-proxy-groups", type=int, default=100, help="Refuse mud-signature fallback above this many groups")
    parser.add_argument("--bootstrap-reps", type=int, default=50, help="50 is a quick exploratory run; use 300+ for more stable intervals if runtime allows.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args()

    raw = load_data(args.path, args.sheet)
    target_col, depth_col = find_column(raw, TARGET_NAME), find_column(raw, DEPTH_NAME)
    depth_numeric = pd.to_numeric(raw[depth_col], errors="coerce")
    if depth_numeric.isna().any():
        raise SystemExit(f"{int(depth_numeric.isna().sum())} missing/non-numeric depth values; review before grouping.")
    if raw[target_col].isna().any():
        raise SystemExit(f"{int(raw[target_col].isna().sum())} missing target labels; review before grouping.")
    df = raw.copy().reset_index(drop=True)
    df[depth_col] = depth_numeric.to_numpy(dtype=float)
    y = severe_flag(df[target_col]).to_numpy(dtype=int)
    if len(df) < 100:
        raise SystemExit(f"Only {len(df)} labeled rows; too small for this evaluation.")
    if y.sum() == 0:
        unique_labels = df[target_col].astype(str).value_counts().head(30).to_dict()
        raise SystemExit("No severe/complete labels recognized. Observed labels: " + repr(unique_labels))

    diag, seg = order_diagnostics(df[depth_col], args.drop)
    groups = seg
    n_segments = int(pd.Series(groups).nunique())
    confident_proxy = (
        diag["share_abs_depth_steps_le_5m"] > 0.80
        and diag["share_non_decreasing_steps"] > 0.80
        and abs(n_segments - args.expect_wells) <= 5
        and n_segments > 1
    )
    print(f"rows_used={len(df):,}; missing_cells={int(df.isna().sum().sum()):,}; exact_duplicates={int(df.duplicated().sum()):,}")
    print(f"critical_labels_recognized={int(y.sum()):,} ({y.mean():.3%})")
    print("row-order diagnostics:", json.dumps(diag, indent=2))
    if confident_proxy:
        grouping_method = "row-order depth-reset segments (proxy only; not confirmed well IDs)"
    elif args.allow_mud_signature_proxy:
        groups, mud_cols = mud_signature_groups(df, target_col)
        n_groups = int(pd.Series(groups).nunique())
        if n_groups > args.max_proxy_groups:
            raise SystemExit(f"Mud-signature fallback produced {n_groups:,} groups > {args.max_proxy_groups}; this can be computationally expensive and is not a substitute for well IDs.")
        grouping_method = f"mud-signature proxy using {mud_cols}; NOT well IDs or days"
        print("WARNING: grouped validation uses a mud-signature proxy, not independent wells or days.")
    else:
        raise SystemExit("Row-order evidence is weak. Refusing to infer wells automatically; obtain real Well IDs or opt into a clearly labelled proxy.")
    group_series = pd.Series(groups)
    pos_groups = int(group_series[y == 1].nunique())
    sizes = group_series.value_counts()
    print(f"groups_used={group_series.nunique()} [{grouping_method}]")
    print(f"groups containing critical rows={pos_groups}; independent positive-group count is more informative than row count alone.")

    results = []
    feature_sets = [("model_with_flow_columns", False), ("model_without_flow_columns", True)]
    for result_name, without_flow in feature_sets:
        cols = numeric_feature_columns(df, target_col, without_flow)
        scores, skipped = loo_probabilities(df, groups, y, cols)
        valid = np.isfinite(scores)
        if not valid.any() or y[valid].sum() == 0:
            print(f"Skipping {result_name}: no scored critical rows; skipped groups={skipped}")
            continue
        res = evaluate_score(result_name, y[valid], scores[valid], groups[valid], args.bootstrap_reps, args.seed)
        res.update({
            "features": [str(c) for c in cols],
            "skipped_groups_due_to_insufficient_training_positives_or_classes": skipped,
            "groups_scored": int(pd.Series(groups[valid]).nunique()),
            "grouping_method": grouping_method,
            "depth_excluded_from_features": True,
        })
        print(f"\n{result_name}: features={len(cols)}; prevalence={res['prevalence']:.4f}; "
              f"AP={res['average_precision_full_oof']:.4f}; group-bootstrap AP median="
              f"{res['average_precision_group_bootstrap_median']:.4f} "
              f"(95% group-CI {res['average_precision_group_bootstrap_ci_2.5']:.4f}–"
              f"{res['average_precision_group_bootstrap_ci_97.5']:.4f}); "
              f"lift={res['average_precision_lift_over_prevalence']:.2f}x")
        for key, values in res["recall_at_alert_budgets"].items():
            print(f"  {key}: median={values['median']:.3f} "
                  f"(95% group-CI {values['ci_2.5']:.3f}–{values['ci_97.5']:.3f})")
        if res["average_precision_group_bootstrap_ci_2.5"] <= res["prevalence"] <= res["average_precision_group_bootstrap_ci_97.5"]:
            print("  CAUTION: prevalence is inside the AP interval; detector skill is not robustly demonstrated.")
        results.append(res)

    flowout_col = next((c for c in df.columns if norm_name(c) == "flowout"), None)
    flowin_col = next((c for c in df.columns if norm_name(c) == "flowin"), None)
    if flowout_col is not None:
        flowout = pd.to_numeric(df[flowout_col], errors="coerce")
        in_range = flowout.notna() & flowout.between(0, 100)
        mask = in_range.to_numpy()
        if mask.sum() and y[mask].sum():
            score = (100.0 - flowout.to_numpy(dtype=float))[mask]
            res = evaluate_score("single-variable heuristic: lower FlowOut (%) ranks as higher concern; exploratory only",
                                 y[mask], score, groups[mask], args.bootstrap_reps, args.seed + 991)
            res.update({
                "features": [str(flowout_col)],
                "grouping_method": grouping_method,
                "interpretation_note": "Uncalibrated single-variable proxy. Confirm FlowOut sensor semantics and operating context with a drilling engineer; not an approved alarm rule.",
            })
            print(f"\n{res['name']}: AP={res['average_precision_full_oof']:.4f}; prevalence={res['prevalence']:.4f}; "
                  f"group-bootstrap median={res['average_precision_group_bootstrap_median']:.4f}; "
                  f"95% CI={res['average_precision_group_bootstrap_ci_2.5']:.4f}–{res['average_precision_group_bootstrap_ci_97.5']:.4f}")
            results.append(res)
    if flowin_col is not None and flowout_col is not None:
        print("\nUnit warning: the published Azadegan table describes FlowIn in gpm and FlowOut in percent. Do NOT compute FlowIn - FlowOut as if units matched.")

    report = {
        "project": "HalaLoya AI Drilling Assistant",
        "task": "Retrospective same-row binary screening: Severe/Complete Loss vs other classes",
        "input_file": str(Path(args.path).name),
        "target_column": str(target_col),
        "depth_column": str(depth_col),
        "raw_rows": int(len(raw)),
        "rows_used": int(len(df)),
        "critical_rows": int(y.sum()),
        "critical_prevalence": float(y.mean()),
        "class_counts": {str(k): int(v) for k, v in df[target_col].astype(str).value_counts().items()},
        "order_diagnostics": diag,
        "grouping_method": grouping_method,
        "group_count": int(group_series.nunique()),
        "groups_with_critical_rows": pos_groups,
        "group_sizes_min_median_max": [int(sizes.min()), float(sizes.median()), int(sizes.max())],
        "results": results,
        "limitations": [
            "Row-order segments are only proxy groups until verified against real well IDs.",
            "Several segments start at substantial depth; a segment count near 20 alone does not prove one segment per well.",
            "Mud signatures are not time/report-day IDs and may group rows from different wells.",
            "M.Depth is excluded from predictors but is used to diagnose possible segment resets.",
            "This is same-row detection of recorded labels, not a future-event early-warning model.",
            "The group bootstrap CI is conditional on generated out-of-group scores and does not capture full training uncertainty.",
            "A proper independent-well test requires real well identifiers or a verified row-to-well mapping.",
            "FlowIn is reported in gpm and FlowOut in percent in the article; subtracting the two directly is invalid."
        ],
    }
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        print(f"\nJSON report written: {args.out_json}")


if __name__ == "__main__":
    main()
