import hashlib
import io
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="HalaLoya AI Drilling Assistant", page_icon="🛢️", layout="wide")

st.title("HalaLoya AI Drilling Assistant")
st.caption("Early research prototype · CSV/Excel exploration and data-quality checks")

st.warning(
    "Research use only. This prototype offers an experimental offline loss-severity classifier, "
    "but it is not a real-time early-warning system, is not validated for drilling operations, "
    "and must not be used for safety-critical decisions. This is a public hosted app: do not upload "
    "confidential or commercially sensitive company data."
)

with st.expander("Before uploading data", expanded=True):
    st.markdown(
        "- Upload only data you are authorized to use.\n"
        "- Do not upload confidential, commercially sensitive, personal, or regulated information to this public prototype.\n"
        "- Verify the dataset license and attribution requirements before research publication.\n"
        "- Results are exploratory statistics, not engineering conclusions."
    )

uploaded = st.file_uploader("Upload a drilling dataset (CSV or Excel XLSX)", type=["csv", "xlsx"])

if uploaded is None:
    st.info("Upload a CSV or XLSX file to inspect its structure and explore numeric variables.")
    st.stop()

raw = uploaded.getvalue()
dataset_fingerprint = hashlib.sha256(raw).hexdigest()
source_sheet = None
try:
    if uploaded.name.lower().endswith(".csv"):
        df = pd.read_csv(io.BytesIO(raw), low_memory=False)
    elif uploaded.name.lower().endswith(".xlsx"):
        workbook = pd.ExcelFile(io.BytesIO(raw), engine="openpyxl")
        if not workbook.sheet_names:
            st.error("No worksheets were found in this Excel workbook.")
            st.stop()
        source_sheet = st.selectbox("Choose an Excel worksheet", workbook.sheet_names)
        df = pd.read_excel(workbook, sheet_name=source_sheet)
    else:
        st.error("Unsupported file type. Upload a CSV or XLSX file.")
        st.stop()
except Exception as exc:
    st.error(f"Could not read this file. Check the file format, encoding, and worksheet structure. Details: {exc}")
    st.stop()

if df.empty:
    st.error("The CSV contains no data rows.")
    st.stop()

st.subheader("Dataset overview")
a, b, c, d = st.columns(4)
a.metric("Rows", f"{len(df):,}")
b.metric("Columns", f"{len(df.columns):,}")
c.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")
missing_total = int(df.isna().sum().sum())
d.metric("Missing cells", f"{missing_total:,}")

st.subheader("Column summary")
summary = pd.DataFrame({
    "column": df.columns.astype(str),
    "data_type": [str(df[col].dtype) for col in df.columns],
    "missing_count": [int(df[col].isna().sum()) for col in df.columns],
    "missing_percent": [round(float(df[col].isna().mean() * 100), 2) for col in df.columns],
    "unique_values": [int(df[col].nunique(dropna=True)) for col in df.columns],
})
st.dataframe(summary, use_container_width=True, hide_index=True)

# If the known label column is present, show its distribution automatically.
target_column = next(
    (
        col for col in df.columns
        if "".join(char.lower() for char in str(col) if char.isalnum()) == "lossesseverity"
    ),
    None,
)
target_distribution_records = None
if target_column is not None:
    target_counts = (
        df[target_column].astype("string").fillna("[Missing]").astype(str).value_counts(dropna=False)
    )
    target_distribution = pd.DataFrame({
        "class": target_counts.index.astype(str),
        "count": target_counts.to_numpy(dtype=int),
        "percent": (target_counts.to_numpy(dtype=float) / len(df) * 100).round(3),
    })
    target_distribution_records = target_distribution.to_dict(orient="records")

    st.subheader("Loss severity class distribution")
    st.caption(
        f"Detected label column: {target_column}. This chart describes the labels in the uploaded data; "
        "it is not model performance and does not show predicted outcomes."
    )
    st.dataframe(target_distribution, use_container_width=True, hide_index=True)
    class_fig = px.bar(
        target_distribution,
        x="class",
        y="count",
        text="percent",
        title="Records by loss-severity class",
    )
    class_fig.update_traces(texttemplate="%{text:.3f}%", textposition="outside", cliponaxis=False)
    class_fig.update_layout(xaxis_title="Loss severity class", yaxis_title="Record count")
    st.plotly_chart(class_fig, use_container_width=True)
    st.warning(
        "Check class imbalance before any modelling. Overall accuracy alone can hide poor "
        "performance on rare severe/complete-loss classes. This chart does not indicate "
        "predictive performance or operational risk."
    )

# Diagnose whether loss-severity classes are distributed unevenly across measured depth.
depth_column_for_audit = next(
    (
        col for col in df.columns
        if "".join(char.lower() for char in str(col) if char.isalnum()) == "mdepth"
    ),
    None,
)
if target_column is not None and depth_column_for_audit is not None:
    depth_audit = df[[depth_column_for_audit, target_column]].copy()
    depth_audit[depth_column_for_audit] = pd.to_numeric(
        depth_audit[depth_column_for_audit], errors="coerce"
    )
    depth_audit = depth_audit.dropna(subset=[depth_column_for_audit, target_column])
    if len(depth_audit) >= 20 and depth_audit[depth_column_for_audit].nunique() >= 3:
        try:
            depth_audit["depth_band"] = pd.qcut(
                depth_audit[depth_column_for_audit],
                q=5,
                duplicates="drop",
            )
            counts_by_depth = pd.crosstab(depth_audit["depth_band"], depth_audit[target_column])
            counts_by_depth["TOTAL"] = counts_by_depth.sum(axis=1)
            percentages_by_depth = pd.crosstab(
                depth_audit["depth_band"],
                depth_audit[target_column],
                normalize="index",
            ).mul(100).round(2)
            st.subheader("Loss severity by measured-depth band")
            st.caption(
                "This descriptive audit groups observed records into up to five depth bands. "
                "It shows where each class appears in this dataset; it is not a causal analysis "
                "or a prediction. A class absent from a depth band cannot be evaluated there."
            )
            st.write("Record counts by depth band")
            st.dataframe(counts_by_depth, use_container_width=True)
            st.write("Within-band class percentages (%)")
            st.dataframe(percentages_by_depth, use_container_width=True)
            depth_plot = percentages_by_depth.reset_index().melt(
                id_vars=["depth_band"],
                var_name="loss_severity",
                value_name="percent",
            )
            depth_fig = px.bar(
                depth_plot,
                x="depth_band",
                y="percent",
                color="loss_severity",
                barmode="stack",
                title="Observed loss-severity mix across measured-depth bands",
            )
            depth_fig.update_layout(
                xaxis_title="Measured-depth band",
                yaxis_title="Records within band (%)",
            )
            st.plotly_chart(depth_fig, use_container_width=True)
        except (ValueError, TypeError) as exc:
            st.info(f"Depth-band audit could not be created for this dataset: {exc}")

st.subheader("Preview")
st.dataframe(df.head(100), use_container_width=True)

numeric_cols = [col for col in df.select_dtypes(include="number").columns.tolist() if col != target_column]
if numeric_cols:
    st.subheader("Exploratory chart")
    x_options = ["Row order"] + [str(col) for col in df.columns]
    x_choice = st.selectbox("X-axis", x_options, index=0)
    y_choice = st.selectbox("Numeric variable (Y-axis)", numeric_cols)
    chart_df = df.copy()
    if x_choice == "Row order":
        chart_df["_row_order_"] = range(1, len(chart_df) + 1)
        x_col = "_row_order_"
        x_label = "Row order (not necessarily time or measured depth)"
    else:
        x_col = x_choice
        x_label = x_choice

    chart_df = chart_df[[x_col, y_choice]].dropna()
    if len(chart_df) > 5000:
        chart_df = chart_df.sample(5000, random_state=42).sort_index()
        st.caption("Showing a reproducible sample of up to 5,000 non-missing rows for responsiveness.")
    fig = px.scatter(chart_df, x=x_col, y=y_choice, title=f"{y_choice} by {x_label}")
    fig.update_layout(xaxis_title=x_label, yaxis_title=str(y_choice))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No numeric columns were detected, so a numeric chart cannot be created.")

st.subheader("Optional statistical screening")
st.caption("IQR flags are simple statistical outliers only. They are not lost-circulation events or risk predictions.")
if numeric_cols:
    selected = st.selectbox("Select a numeric column", numeric_cols, key="iqr_col")
    values = pd.to_numeric(df[selected], errors="coerce").dropna()
    if len(values) >= 4:
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        flagged = ((values < lower) | (values > upper))
        st.write(f"IQR bounds: **{lower:.6g}** to **{upper:.6g}**")
        st.write(f"Values flagged by this rule: **{int(flagged.sum()):,}** of {len(values):,}")
    else:
        st.info("At least four non-missing numeric values are needed for this screening.")

# Optional exploratory classifier. This is not an operational prediction tool.
EVALUATION_SCHEMA_VERSION = 12
baseline_results = st.session_state.get("baseline_eval_results")
if baseline_results and (
    baseline_results.get("dataset_fingerprint") != dataset_fingerprint
    or baseline_results.get("validation_method") != st.session_state.get("baseline_split_method")
    or baseline_results.get("evaluation_schema_version") != EVALUATION_SCHEMA_VERSION
    or "train_class_counts" not in baseline_results
    or "test_class_counts" not in baseline_results
    or "predicted_class_counts" not in baseline_results
):
    st.session_state.pop("baseline_eval_results", None)
    baseline_results = None

if target_column is not None:
    st.subheader("Exploratory model comparison (research only)")
    split_method = st.radio(
        "Choose an evaluation split",
        [
            "Random stratified row split",
            "Depth-block stress test (lower depths → higher depths)",
        ],
        key="baseline_split_method",
    )
    if split_method == "Random stratified row split":
        st.warning(
            "Random row-level split: records from the same well or nearby depths may appear "
            "in both sets. These metrics do not prove performance on unseen wells or other "
            "fields. Do not use predictions for operational decisions."
        )
    else:
        st.warning(
            "Depth-block stress test: compare candidates using an internal validation subset "
            "of lower-depth training data, then evaluate the selected candidate on the higher-depth "
            "block. The same wells may still appear in both partitions; this is not a well-held-out "
            "test. Do not use predictions for operational decisions."
        )
    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    depth_column_for_split = next(
        (
            col for col in numeric_columns
            if "".join(char.lower() for char in str(col) if char.isalnum()) == "mdepth"
        ),
        None,
    )
    feature_columns = [
        col for col in numeric_columns
        if col != target_column
        and "".join(char.lower() for char in str(col) if char.isalnum()) != "mdepth"
    ]
    st.caption(
        f"Target: {target_column}. Numeric input features available: {len(feature_columns)}. "
        "The target and M.Depth columns are excluded from model inputs in this evaluation; "
        "depth is used only for depth-band audits and, when selected, to define the depth-block split."
    )
    if len(feature_columns) < 2:
        st.error("At least two numeric input features are required for this experimental baseline.")
    else:
        if st.button("Compare models and evaluate selected candidate", key="run_baseline_model"):
            try:
                usable_columns = list(feature_columns) + [target_column]
                if depth_column_for_split is not None and depth_column_for_split not in usable_columns:
                    usable_columns.append(depth_column_for_split)
                usable = df.loc[df[target_column].notna(), usable_columns].copy()
                for col in feature_columns:
                    usable[col] = pd.to_numeric(usable[col], errors="coerce")
                usable[feature_columns] = usable[feature_columns].replace([np.inf, -np.inf], np.nan)
                before_drop = len(usable)
                usable = usable.dropna(subset=feature_columns + [target_column])
                dropped_rows = before_drop - len(usable)
                X = usable[feature_columns]
                y = usable[target_column].astype(str)
                counts = y.value_counts()
                if len(usable) < 50 or y.nunique() < 2:
                    st.error("There are not enough labeled records/classes for this experiment.")
                elif split_method == "Random stratified row split" and (counts < 2).any():
                    st.error(
                        "At least one target class has fewer than two usable records; "
                        "a stratified train/test split cannot be performed reliably."
                    )
                else:
                    class_names = sorted(y.unique().tolist())
                    depth_cutoff = None
                    if split_method == "Random stratified row split":
                        X_train, X_test, y_train, y_test = train_test_split(
                            X,
                            y,
                            test_size=0.30,
                            random_state=42,
                            stratify=y,
                        )
                        validation_limitation = (
                            "Random row-level split; no well-level holdout because the visible "
                            "columns do not identify wells. Metrics are exploratory only."
                        )
                    else:
                        depth_column = depth_column_for_split
                        if depth_column is None:
                            st.error(
                                "The depth-block stress test requires a numeric M.Depth column."
                            )
                            st.stop()
                        unique_depths = np.sort(pd.to_numeric(usable[depth_column], errors="coerce").dropna().unique())
                        if len(unique_depths) < 5:
                            st.error("There are too few unique depth values for a depth-block test.")
                            st.stop()
                        depth_index = min(max(int(len(unique_depths) * 0.70), 1), len(unique_depths) - 1)
                        depth_cutoff = float(unique_depths[depth_index])
                        train_mask = pd.to_numeric(usable[depth_column], errors="coerce") < depth_cutoff
                        test_mask = ~train_mask
                        X_train, X_test = X.loc[train_mask], X.loc[test_mask]
                        y_train, y_test = y.loc[train_mask], y.loc[test_mask]
                        if len(y_train) < 50 or len(y_test) < 10 or y_train.nunique() < 2:
                            st.error(
                                "The depth-block split produced too few rows or fewer than two "
                                "training classes; this dataset cannot support this split as configured."
                            )
                            st.stop()
                        validation_limitation = (
                            "Depth-block sensitivity test: train on measured depths below the "
                            "70th-percentile unique-depth cutoff and test on depths at/above it. "
                            "It is not a well-held-out test; wells may still overlap across partitions."
                        )
                    # Select among candidates using only a validation subset of the outer training rows.
                    # The outer test set is kept separate from model selection.
                    train_counts_for_validation = y_train.value_counts()
                    if (train_counts_for_validation < 2).any():
                        st.error(
                            "Model comparison needs at least two training examples per class "
                            "to form the internal stratified validation split."
                        )
                        st.stop()

                    inner_train_X, validation_X, inner_train_y, validation_y = train_test_split(
                        X_train,
                        y_train,
                        test_size=0.20,
                        random_state=24,
                        stratify=y_train,
                    )

                    candidate_specs = [
                        (
                            "Random Forest · balanced_subsample · leaf=2",
                            RandomForestClassifier(
                                n_estimators=100,
                                min_samples_leaf=2,
                                class_weight="balanced_subsample",
                                random_state=42,
                                n_jobs=-1,
                            ),
                            False,
                        ),
                        (
                            "Random Forest · balanced · leaf=1",
                            RandomForestClassifier(
                                n_estimators=150,
                                min_samples_leaf=1,
                                class_weight="balanced",
                                random_state=43,
                                n_jobs=-1,
                            ),
                            False,
                        ),
                        (
                            "Extra Trees · balanced · leaf=1",
                            ExtraTreesClassifier(
                                n_estimators=150,
                                min_samples_leaf=1,
                                class_weight="balanced",
                                random_state=44,
                                n_jobs=-1,
                            ),
                            False,
                        ),
                        (
                            "Decision Tree · balanced · leaf=2",
                            DecisionTreeClassifier(
                                class_weight="balanced",
                                min_samples_leaf=2,
                                random_state=45,
                            ),
                            False,
                        ),
                        (
                            "AdaBoost · shallow tree · balanced sample weights",
                            AdaBoostClassifier(
                                estimator=DecisionTreeClassifier(max_depth=2, random_state=46),
                                n_estimators=80,
                                learning_rate=0.5,
                                random_state=46,
                            ),
                            True,
                        ),
                    ]

                    candidate_validation_results = []
                    for candidate_name, candidate_model, use_sample_weights in candidate_specs:
                        if use_sample_weights:
                            candidate_weights = compute_sample_weight(
                                class_weight="balanced",
                                y=inner_train_y,
                            )
                            candidate_model.fit(
                                inner_train_X,
                                inner_train_y,
                                sample_weight=candidate_weights,
                            )
                        else:
                            candidate_model.fit(inner_train_X, inner_train_y)
                        validation_predictions = candidate_model.predict(validation_X)
                        val_precision, val_recall, val_f1, val_support = precision_recall_fscore_support(
                            validation_y,
                            validation_predictions,
                            labels=class_names,
                            zero_division=0,
                        )
                        candidate_validation_results.append({
                            "candidate": candidate_name,
                            "validation_accuracy": round(float(accuracy_score(validation_y, validation_predictions)), 4),
                            "validation_balanced_accuracy": round(float(balanced_accuracy_score(validation_y, validation_predictions)), 4),
                            "validation_macro_f1": round(float(f1_score(
                                validation_y,
                                validation_predictions,
                                labels=class_names,
                                average="macro",
                                zero_division=0,
                            )), 4),
                            "validation_per_class_metrics": [
                                {
                                    "class": class_names[i],
                                    "precision": round(float(val_precision[i]), 4),
                                    "recall": round(float(val_recall[i]), 4),
                                    "f1": round(float(val_f1[i]), 4),
                                    "validation_support": int(val_support[i]),
                                }
                                for i in range(len(class_names))
                            ],
                        })

                    best_candidate_index = max(
                        range(len(candidate_specs)),
                        key=lambda i: (
                            candidate_validation_results[i]["validation_macro_f1"],
                            candidate_validation_results[i]["validation_balanced_accuracy"],
                        ),
                    )
                    selected_model_name, model, selected_use_sample_weights = candidate_specs[best_candidate_index]
                    selected_validation_metrics = candidate_validation_results[best_candidate_index]

                    # Refit the selected candidate on all outer training rows, then evaluate
                    # once on the held-out test data.
                    if selected_use_sample_weights:
                        training_weights = compute_sample_weight(class_weight="balanced", y=y_train)
                        model.fit(X_train, y_train, sample_weight=training_weights)
                    else:
                        model.fit(X_train, y_train)
                    majority = DummyClassifier(strategy="most_frequent")
                    majority.fit(X_train, y_train)
                    majority_pred = majority.predict(X_test)
                    predictions = model.predict(X_test)

                    # Separate critical-loss screening task: Severe/Complete vs all other labels.
                    # Threshold/model selection uses inner validation only; final metrics use outer test.
                    critical_pattern = r"severe|complete"
                    critical_train_inner = inner_train_y.astype(str).str.contains(
                        critical_pattern, case=False, regex=True
                    ).astype(int)
                    critical_validation_y = validation_y.astype(str).str.contains(
                        critical_pattern, case=False, regex=True
                    ).astype(int)
                    critical_train_y = y_train.astype(str).str.contains(
                        critical_pattern, case=False, regex=True
                    ).astype(int)
                    critical_test_y = y_test.astype(str).str.contains(
                        critical_pattern, case=False, regex=True
                    ).astype(int)

                    critical_candidate_specs = [
                        (
                            "Random Forest · balanced_subsample",
                            RandomForestClassifier(
                                n_estimators=150,
                                min_samples_leaf=1,
                                class_weight="balanced_subsample",
                                random_state=71,
                                n_jobs=-1,
                            ),
                        ),
                        (
                            "Extra Trees · balanced",
                            ExtraTreesClassifier(
                                n_estimators=150,
                                min_samples_leaf=1,
                                class_weight="balanced",
                                random_state=72,
                                n_jobs=-1,
                            ),
                        ),
                        (
                            "Decision Tree · balanced",
                            DecisionTreeClassifier(
                                class_weight="balanced",
                                min_samples_leaf=3,
                                random_state=73,
                            ),
                        ),
                    ]

                    critical_candidate_validation = []
                    if critical_train_inner.nunique() < 2 or critical_validation_y.nunique() < 2:
                        critical_screen_results = {
                            "available": False,
                            "reason": (
                                "The internal training/validation split does not contain both "
                                "critical and non-critical examples."
                            ),
                        }
                    else:
                        thresholds = np.round(np.arange(0.05, 1.00, 0.05), 2)
                        for critical_name, critical_model in critical_candidate_specs:
                            critical_model.fit(
                                inner_train_X,
                                critical_train_inner,
                            )
                            critical_val_prob = critical_model.predict_proba(validation_X)[
                                :, list(critical_model.classes_).index(1)
                            ]
                            best_threshold = 0.50
                            best_validation_metrics = None
                            for threshold in thresholds:
                                val_binary_pred = (critical_val_prob >= threshold).astype(int)
                                precision_pos, recall_pos, f1_pos, _ = precision_recall_fscore_support(
                                    critical_validation_y,
                                    val_binary_pred,
                                    labels=[1],
                                    average=None,
                                    zero_division=0,
                                )
                                f2_pos = fbeta_score(
                                    critical_validation_y,
                                    val_binary_pred,
                                    beta=2,
                                    zero_division=0,
                                )
                                selection_metrics = {
                                    "threshold": float(threshold),
                                    "validation_f2": float(f2_pos),
                                    "validation_precision": float(precision_pos[0]),
                                    "validation_recall": float(recall_pos[0]),
                                    "validation_f1": float(f1_pos[0]),
                                }
                                selection_key = (
                                    selection_metrics["validation_f2"],
                                    selection_metrics["validation_recall"],
                                    selection_metrics["validation_precision"],
                                    selection_metrics["threshold"],
                                )
                                if (
                                    best_validation_metrics is None
                                    or selection_key > best_validation_metrics["selection_key"]
                                ):
                                    best_validation_metrics = {
                                        **selection_metrics,
                                        "selection_key": selection_key,
                                    }
                                    best_threshold = float(threshold)
                            critical_candidate_validation.append({
                                "candidate": critical_name,
                                "selected_threshold": best_threshold,
                                "validation_precision": round(best_validation_metrics["validation_precision"], 4),
                                "validation_recall": round(best_validation_metrics["validation_recall"], 4),
                                "validation_f1": round(best_validation_metrics["validation_f1"], 4),
                                "validation_f2": round(best_validation_metrics["validation_f2"], 4),
                                "_selection_key": best_validation_metrics["selection_key"],
                            })

                        critical_candidate_validation.sort(
                            key=lambda item: item["_selection_key"],
                            reverse=True,
                        )
                        best_critical_candidate = critical_candidate_validation[0]
                        selected_critical_name = best_critical_candidate["candidate"]
                        selected_critical_threshold = best_critical_candidate["selected_threshold"]
                        selected_critical_model = next(
                            candidate for name, candidate in critical_candidate_specs
                            if name == selected_critical_name
                        )
                        selected_critical_model.fit(X_train, critical_train_y)
                        critical_test_prob = selected_critical_model.predict_proba(X_test)[
                            :, list(selected_critical_model.classes_).index(1)
                        ]
                        critical_test_pred = (
                            critical_test_prob >= selected_critical_threshold
                        ).astype(int)
                        positive_test_scores = critical_test_prob[critical_test_y.to_numpy() == 1]
                        negative_test_scores = critical_test_prob[critical_test_y.to_numpy() == 0]
                        test_average_precision = (
                            float(average_precision_score(critical_test_y, critical_test_prob))
                            if critical_test_y.nunique() > 1 and int(critical_test_y.sum()) > 0
                            else None
                        )
                        max_positive_score = (
                            float(np.max(positive_test_scores)) if len(positive_test_scores) else None
                        )
                        median_positive_score = (
                            float(np.median(positive_test_scores)) if len(positive_test_scores) else None
                        )
                        max_negative_score = (
                            float(np.max(negative_test_scores)) if len(negative_test_scores) else None
                        )
                        median_negative_score = (
                            float(np.median(negative_test_scores)) if len(negative_test_scores) else None
                        )
                        critical_cm = confusion_matrix(
                            critical_test_y,
                            critical_test_pred,
                            labels=[0, 1],
                        )
                        tn, fp, fn, tp = [int(v) for v in critical_cm.ravel()]
                        critical_precision, critical_recall, critical_f1, _ = precision_recall_fscore_support(
                            critical_test_y,
                            critical_test_pred,
                            labels=[1],
                            average=None,
                            zero_division=0,
                        )
                        critical_screen_results = {
                            "available": True,
                            "task": "Severe Loss or Complete Loss vs all other classes",
                            "selected_model": selected_critical_name,
                            "threshold_selected_on_internal_validation": selected_critical_threshold,
                            "validation_metric_for_selection": "F2 (recall-weighted), tie-break by recall, then precision, then higher threshold",
                            "candidate_validation_results": [
                                {k: v for k, v in item.items() if not k.startswith("_")}
                                for item in critical_candidate_validation
                            ],
                            "training_critical_rows": int(critical_train_y.sum()),
                            "test_critical_rows": int(critical_test_y.sum()),
                            "predicted_critical_rows": int(critical_test_pred.sum()),
                            "test_precision_critical": round(float(critical_precision[0]), 4),
                            "test_recall_critical": round(float(critical_recall[0]), 4),
                            "test_f1_critical": round(float(critical_f1[0]), 4),
                            "test_f2_critical": round(float(fbeta_score(critical_test_y, critical_test_pred, beta=2, zero_division=0)), 4),
                            "test_average_precision": round(test_average_precision, 4) if test_average_precision is not None else None,
                            "max_test_critical_probability": round(max_positive_score, 6) if max_positive_score is not None else None,
                            "median_test_critical_probability": round(median_positive_score, 6) if median_positive_score is not None else None,
                            "max_test_noncritical_probability": round(max_negative_score, 6) if max_negative_score is not None else None,
                            "median_test_noncritical_probability": round(median_negative_score, 6) if median_negative_score is not None else None,
                            "test_specificity": round(float(tn / (tn + fp)), 4) if (tn + fp) else None,
                            "true_negatives": tn,
                            "false_positives": fp,
                            "false_negatives": fn,
                            "true_positives": tp,
                            "confusion_matrix_labels": ["Other severity", "Severe/Complete Loss"],
                            "validation_limitation": (
                                "Binary critical-loss screening on recorded class labels, not a future-event early-warning model. "
                                "The depth-block test has no Complete Loss test examples, so the result measures high-severity "
                                "screening only on critical classes actually present in this test block."
                            ),
                        }

                    # Hard consistency checks: every test row must have one true label and one prediction.
                    train_class_counts = {str(k): int(v) for k, v in y_train.value_counts().items()}
                    test_class_counts = {str(k): int(v) for k, v in y_test.value_counts().items()}
                    predicted_class_counts = {
                        str(k): int(v) for k, v in pd.Series(predictions).value_counts().items()
                    }
                    actual_train_total = sum(train_class_counts.values())
                    actual_test_total = sum(test_class_counts.values())
                    predicted_total = sum(predicted_class_counts.values())
                    if (
                        actual_train_total != len(y_train)
                        or actual_test_total != len(y_test)
                        or predicted_total != len(y_test)
                        or len(predictions) != len(y_test)
                    ):
                        st.error(
                            "Evaluation counts are inconsistent; no metrics were saved. "
                            f"Training rows: {len(y_train):,} vs class total {actual_train_total:,}; "
                            f"test rows: {len(y_test):,} vs class total {actual_test_total:,}; "
                            f"predictions: {len(predictions):,} vs test rows {len(y_test):,}."
                        )
                        st.stop()

                    precision, recall, f1_values, support = precision_recall_fscore_support(
                        y_test,
                        predictions,
                        labels=class_names,
                        zero_division=0,
                    )
                    cm = confusion_matrix(y_test, predictions, labels=class_names)
                    per_class = [
                        {
                            "class": class_names[i],
                            "precision": round(float(precision[i]), 4),
                            "recall": round(float(recall[i]), 4),
                            "f1": round(float(f1_values[i]), 4),
                            "test_support": int(support[i]),
                        }
                        for i in range(len(class_names))
                    ]
                    top_features = sorted(
                        [
                            {"feature": str(feature_columns[i]), "importance": round(float(model.feature_importances_[i]), 6)}
                            for i in range(len(feature_columns))
                        ],
                        key=lambda item: item["importance"],
                        reverse=True,
                    )[:12]
                    st.session_state["baseline_eval_results"] = {
                        "dataset_fingerprint": dataset_fingerprint,
                        "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
                        "model": selected_model_name,
                        "selected_model_validation_metrics": selected_validation_metrics,
                        "candidate_validation_results": candidate_validation_results,
                        "critical_loss_screen": critical_screen_results,
                        "model_selection_rule": "Highest internal validation Macro F1 over the known class list; tie-breaker is balanced accuracy. Five candidates compared.",
                        "baseline": "DummyClassifier(strategy='most_frequent')",
                        "random_state": 42,
                        "validation_method": split_method,
                        "test_size": round(float(len(y_test) / len(usable)), 6),
                        "depth_cutoff": depth_cutoff,
                        "training_rows": int(len(y_train)),
                        "rows_used": int(len(usable)),
                        "rows_dropped_for_missing_or_non_numeric_values": int(dropped_rows),
                        "feature_columns": [str(col) for col in feature_columns],
                        "class_names": class_names,
                        "class_counts": {str(k): int(v) for k, v in counts.items()},
                        "train_class_counts": train_class_counts,
                        "test_class_counts": test_class_counts,
                        "predicted_class_counts": predicted_class_counts,
                        "training_rows": int(len(y_train)),
                        "test_rows": int(len(y_test)),
                        "majority_baseline_accuracy": round(float(accuracy_score(y_test, majority_pred)), 4),
                        "selected_model_accuracy": round(float(accuracy_score(y_test, predictions)), 4),
                        "balanced_accuracy": round(float(balanced_accuracy_score(y_test, predictions)), 4),
                        "balanced_accuracy_scope": "Classes represented in y_test; classes with zero true test support cannot be evaluated.",
                        "macro_f1": round(float(f1_score(
                            y_test,
                            predictions,
                            labels=class_names,
                            average="macro",
                            zero_division=0,
                        )), 4),
                        "macro_f1_all_known_classes": round(float(f1_score(
                            y_test,
                            predictions,
                            labels=class_names,
                            average="macro",
                            zero_division=0,
                        )), 4),
                        "macro_f1_test_present_classes": round(float(f1_score(
                            y_test,
                            predictions,
                            average="macro",
                            zero_division=0,
                        )), 4),
                        "per_class_metrics": per_class,
                        "confusion_matrix": cm.tolist(),
                        "top_feature_importances": top_features,
                        "validation_limitation": validation_limitation,
                    }
                    baseline_results = st.session_state["baseline_eval_results"]
            except Exception as exc:
                st.error(f"The experimental baseline could not run. Details: {exc}")

        baseline_results = st.session_state.get("baseline_eval_results")
        if baseline_results and baseline_results.get("dataset_fingerprint") == dataset_fingerprint:
            st.caption(
                f"Evaluation used {baseline_results['rows_used']:,} records; "
                f"{baseline_results['test_rows']:,} were in the holdout test set. "
                f"Rows excluded for missing/non-numeric values: "
                f"{baseline_results['rows_dropped_for_missing_or_non_numeric_values']:,}."
            )
            st.caption(
                f"Selected candidate: {baseline_results['model']}. "
                "Candidates were compared on an internal validation subset of the training data; "
                "the selected model was refit on all training rows before the held-out test."
            )
            candidate_table = pd.DataFrame([
                {
                    "candidate": item["candidate"],
                    "validation_accuracy": item["validation_accuracy"],
                    "validation_balanced_accuracy": item["validation_balanced_accuracy"],
                    "validation_macro_f1": item["validation_macro_f1"],
                }
                for item in baseline_results["candidate_validation_results"]
            ])
            st.subheader("Candidate comparison on training-only validation data")
            st.dataframe(candidate_table, use_container_width=True, hide_index=True)

            critical_screen = baseline_results.get("critical_loss_screen", {})
            st.subheader("Critical-loss screening experiment (Severe/Complete vs other classes)")
            st.warning(
                "This separate binary classifier tries to identify records labelled Severe Loss or Complete Loss. "
                "It is not a real-time early-warning model; it classifies recorded labels using the same-row inputs. "
                "Lowering the threshold may improve recall but can increase false alarms."
            )
            if not critical_screen.get("available", False):
                st.error(critical_screen.get("reason", "Critical-loss screening is unavailable for this split."))
            else:
                critical_candidate_table = pd.DataFrame(critical_screen["candidate_validation_results"])
                st.caption(
                    "Critical-screen candidates and thresholds were selected using internal validation data only. "
                    "The final row of metrics below is measured on the held-out test block."
                )
                st.dataframe(critical_candidate_table, use_container_width=True, hide_index=True)
                st.caption(
                    f"Selected: {critical_screen['selected_model']} at probability threshold "
                    f"{critical_screen['threshold_selected_on_internal_validation']:.2f}"
                )
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Critical-loss precision", f"{critical_screen['test_precision_critical']:.1%}")
                c2.metric("Critical-loss recall", f"{critical_screen['test_recall_critical']:.1%}")
                c3.metric("Critical-loss F1", f"{critical_screen['test_f1_critical']:.3f}")
                c4.metric("Critical-loss F2", f"{critical_screen['test_f2_critical']:.3f}")
                d1, d2 = st.columns(2)
                d1.metric(
                    "Test average precision (critical class)",
                    f"{critical_screen['test_average_precision']:.3f}"
                    if critical_screen.get("test_average_precision") is not None else "N/A",
                )
                d2.metric(
                    "Selected threshold (chosen on validation only)",
                    f"{critical_screen['threshold_selected_on_internal_validation']:.3f}",
                )
                score_diag = pd.DataFrame([
                    {
                        "score summary": "Actual critical cases in test",
                        "max predicted critical probability": critical_screen.get("max_test_critical_probability"),
                        "median predicted critical probability": critical_screen.get("median_test_critical_probability"),
                    },
                    {
                        "score summary": "Actual non-critical cases in test",
                        "max predicted critical probability": critical_screen.get("max_test_noncritical_probability"),
                        "median predicted critical probability": critical_screen.get("median_test_noncritical_probability"),
                    },
                ])
                st.caption(
                    "Score diagnostics help distinguish a threshold issue from poor ranking on the deeper test block. "
                    "The test scores are diagnostic only; do not retune the threshold using them."
                )
                st.dataframe(score_diag, use_container_width=True, hide_index=True)
                c5, c6, c7, c8 = st.columns(4)
                c5.metric("True positives", f"{critical_screen['true_positives']:,}")
                c6.metric("Missed critical cases", f"{critical_screen['false_negatives']:,}")
                c7.metric("False alarms", f"{critical_screen['false_positives']:,}")
                c8.metric("Specificity", f"{critical_screen['test_specificity']:.1%}" if critical_screen["test_specificity"] is not None else "N/A")
                critical_cm_df = pd.DataFrame(
                    [
                        [critical_screen["true_negatives"], critical_screen["false_positives"]],
                        [critical_screen["false_negatives"], critical_screen["true_positives"]],
                    ],
                    index=["Actual: Other severity", "Actual: Severe/Complete"],
                    columns=["Predicted: Other severity", "Predicted: Severe/Complete"],
                )
                st.dataframe(critical_cm_df, use_container_width=True, hide_index=False)
                st.caption(
                    f"Test support — critical cases: {critical_screen['test_critical_rows']:,}; "
                    f"predicted critical cases: {critical_screen['predicted_critical_rows']:,}. "
                    "Complete Loss has no true examples in the high-depth test block, so that class cannot be evaluated independently."
                )

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Majority baseline accuracy", f"{baseline_results['majority_baseline_accuracy']:.1%}")
            m2.metric("Selected model accuracy", f"{baseline_results['selected_model_accuracy']:.1%}")
            m3.metric("Balanced accuracy (classes present in test)", f"{baseline_results['balanced_accuracy']:.1%}")
            m4.metric("Macro F1 (all known classes)", f"{baseline_results['macro_f1_all_known_classes']:.3f}")
            st.metric(
                "Macro F1 (classes present in test only)",
                f"{baseline_results['macro_f1_test_present_classes']:.3f}",
            )
            st.caption(
                "The all-known-classes Macro F1 includes all target classes, even classes with no true test examples "
                "(for which this strict score assigns zero F1). The test-present Macro F1 averages only classes "
                "represented in the test labels or predictions. Check test support before interpreting either score."
            )
            support_audit = pd.DataFrame({
                "class": baseline_results["class_names"],
                "training_rows": [
                    baseline_results["train_class_counts"].get(c, 0)
                    for c in baseline_results["class_names"]
                ],
                "test_rows": [
                    baseline_results["test_class_counts"].get(c, 0)
                    for c in baseline_results["class_names"]
                ],
                "predicted_rows": [
                    baseline_results["predicted_class_counts"].get(c, 0)
                    for c in baseline_results["class_names"]
                ],
            })
            support_totals = {
                "training_rows": int(support_audit["training_rows"].sum()),
                "test_rows": int(support_audit["test_rows"].sum()),
                "predicted_rows": int(support_audit["predicted_rows"].sum()),
            }
            support_audit = pd.concat(
                [
                    support_audit,
                    pd.DataFrame([{
                        "class": "TOTAL",
                        "training_rows": support_totals["training_rows"],
                        "test_rows": support_totals["test_rows"],
                        "predicted_rows": support_totals["predicted_rows"],
                    }]),
                ],
                ignore_index=True,
            )
            st.caption("Class support audit: totals must match training rows, test rows, and number of predictions.")
            st.dataframe(support_audit, use_container_width=True, hide_index=True)
            counts_ok = (
                support_totals["training_rows"] == baseline_results["training_rows"]
                and support_totals["test_rows"] == baseline_results["test_rows"]
                and support_totals["predicted_rows"] == baseline_results["test_rows"]
            )
            if not counts_ok:
                st.error(
                    "These saved results fail the count-consistency check. Do not interpret the metrics; "
                    "refresh the app and rerun the experiment."
                )
            absent_test_classes = support_audit.loc[support_audit["test_rows"] == 0, "class"].tolist()
            unpredicted_classes = support_audit.loc[support_audit["predicted_rows"] == 0, "class"].tolist()
            if absent_test_classes:
                st.warning(
                    "The test set contains no true examples of: "
                    + ", ".join(absent_test_classes)
                    + ". Recall for these classes cannot be evaluated on this split."
                )
            if unpredicted_classes:
                st.error(
                    "The model predicted zero examples for: "
                    + ", ".join(unpredicted_classes)
                    + ". Inspect these classes carefully; overall accuracy can hide this failure."
                )
            st.dataframe(
                pd.DataFrame(baseline_results["per_class_metrics"]),
                use_container_width=True,
                hide_index=True,
            )
            class_names = baseline_results["class_names"]
            cm_df = pd.DataFrame(
                baseline_results["confusion_matrix"],
                index=[f"Actual: {name}" for name in class_names],
                columns=[f"Predicted: {name}" for name in class_names],
            )
            st.caption("Confusion matrix: rows are actual classes; columns are predicted classes.")
            st.dataframe(cm_df, use_container_width=True)
            importance_df = pd.DataFrame(baseline_results["top_feature_importances"])
            if not importance_df.empty:
                importance_fig = px.bar(
                    importance_df.sort_values("importance"),
                    x="importance",
                    y="feature",
                    orientation="h",
                    title="Top feature importances in the selected model",
                )
                st.plotly_chart(importance_fig, use_container_width=True)
                st.caption(
                    "Feature importance is a model-specific association measure, not proof "
                    "of causation or engineering relevance."
                )
            st.warning(
                "Interpret each class metric using its test support. The complete-loss class "
                "has very few examples, so its score can change sharply with a small number "
                "of errors. Neither split proves generalization to unseen wells."
            )
            st.caption(f"Evaluation method: {baseline_results.get('validation_method', 'not recorded')}")
            if baseline_results.get("depth_cutoff") is not None:
                st.caption(
                    f"Depth-block cutoff: M.Depth = {baseline_results['depth_cutoff']:g}; "
                    "lower depths were used for training and depths at/above the cutoff for testing."
                )

            # Put the evaluation download next to the model results for easy access.
            model_evaluation_report = {
                "project": "HalaLoya AI Drilling Assistant",
                "report_generated_utc": datetime.now(timezone.utc).isoformat(),
                "uploaded_filename": uploaded.name,
                "source_worksheet": source_sheet,
                "dataset_sha256": dataset_fingerprint,
                "row_count": int(len(df)),
                "column_count": int(len(df.columns)),
                "target_column": str(target_column),
                "target_class_distribution": target_distribution_records,
                "exploratory_model_baseline": {
                    k: v for k, v in baseline_results.items()
                    if k != "dataset_fingerprint"
                },
                "limitations": [
                    "Experimental classification baseline only; not an operational prediction tool.",
                    baseline_results.get("validation_limitation", "Validation limitation not recorded."),
                    "Metrics do not prove generalization to unseen wells, operators, formations, or fields.",
                    "Rare severe/complete-loss class metrics may be unstable due to low support.",
                    "Not validated for drilling operations or safety-critical decisions."
                ]
            }
            st.download_button(
                "Download model evaluation report (JSON)",
                data=json.dumps(model_evaluation_report, ensure_ascii=False, indent=2).encode("utf-8"),
                file_name="halaloya_model_evaluation_report.json",
                mime="application/json",
                key="download_model_evaluation_report",
            )

report = {
    "project": "HalaLoya AI Drilling Assistant",
    "report_generated_utc": datetime.now(timezone.utc).isoformat(),
    "uploaded_filename": uploaded.name,
    "source_worksheet": source_sheet,
    "row_count": int(len(df)),
    "column_count": int(len(df.columns)),
    "duplicate_row_count": int(df.duplicated().sum()),
    "missing_cell_count": missing_total,
    "target_column": str(target_column) if target_column is not None else None,
    "target_class_distribution": target_distribution_records,
    "exploratory_model_baseline": (
        {k: v for k, v in st.session_state["baseline_eval_results"].items() if k != "dataset_fingerprint"}
        if st.session_state.get("baseline_eval_results")
        and st.session_state["baseline_eval_results"].get("dataset_fingerprint") == dataset_fingerprint
        else None
    ),
    "columns": summary.to_dict(orient="records"),
    "limitations": [
        "Exploratory data-quality checks and an optional experimental offline classification baseline only.",
        "The model is evaluated on a random row-level holdout; this does not establish generalization to unseen wells.",
        "The experiment is not a real-time early-warning system and is not validated for operations.",
        "Rare severe/complete-loss metrics may be unstable due to low test support.",
        "Dataset license and permission must be verified independently."
    ]
}
st.download_button(
    "Download data-quality report (JSON)",
    data=json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8"),
    file_name="halaloya_data_quality_report.json",
    mime="application/json",
)
