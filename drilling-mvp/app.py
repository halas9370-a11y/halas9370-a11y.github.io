import hashlib
import io
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
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
EVALUATION_SCHEMA_VERSION = 5
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
    feature_columns = [
        col for col in df.select_dtypes(include="number").columns.tolist()
        if col != target_column
    ]
    st.caption(
        f"Target: {target_column}. Numeric input features available: {len(feature_columns)}. "
        "The target column is excluded from the model inputs."
    )
    if len(feature_columns) < 2:
        st.error("At least two numeric input features are required for this experimental baseline.")
    else:
        if st.button("Compare models and evaluate selected candidate", key="run_baseline_model"):
            try:
                usable = df.loc[df[target_column].notna(), feature_columns + [target_column]].copy()
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
                        depth_column = next(
                            (
                                col for col in feature_columns
                                if "".join(char.lower() for char in str(col) if char.isalnum()) == "mdepth"
                            ),
                            None,
                        )
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
                        ),
                    ]

                    candidate_validation_results = []
                    for candidate_name, candidate_model in candidate_specs:
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
                    selected_model_name, model = candidate_specs[best_candidate_index]
                    selected_validation_metrics = candidate_validation_results[best_candidate_index]

                    # Refit the selected candidate on all outer training rows, then evaluate
                    # once on the held-out test data.
                    model.fit(X_train, y_train)
                    majority = DummyClassifier(strategy="most_frequent")
                    majority.fit(X_train, y_train)
                    majority_pred = majority.predict(X_test)
                    predictions = model.predict(X_test)

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
                        "model_selection_rule": "Highest internal validation Macro F1; tie-breaker is balanced accuracy.",
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
                        "random_forest_accuracy": round(float(accuracy_score(y_test, predictions)), 4),
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
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Majority baseline accuracy", f"{baseline_results['majority_baseline_accuracy']:.1%}")
            m2.metric("Random forest accuracy", f"{baseline_results['random_forest_accuracy']:.1%}")
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
                    title="Top Random Forest feature importances",
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
