# HalaLoya AI Drilling Assistant — Initial Model Evaluation Plan

## Purpose and status

This document records a cautious plan for a research-only first supervised-learning experiment using the publicly published Azadegan oilfield lost-circulation dataset. The deployed app currently inspects CSV/XLSX files and may display the observed `LossesSeverity` class distribution. It does **not** train a model or generate predictions.

## Dataset profile reported for the supplied supplementary workbook

- Rows: 65,376
- Columns: 18 (17 candidate input variables and `LossesSeverity`)
- Duplicate rows reported by the app: 0
- Missing cells reported by the app: 0
- Source article: Mardanirad, Wood & Zakeri (2021), https://doi.org/10.1007/s42452-021-04769-0
- Related class-imbalance discussion: Wood, Mardanirad & Zakeri (2022), https://doi.org/10.1007/s13202-021-01411-y

The counts below were supplied from the workbook review and should be re-computed directly from the raw workbook before a formal result is published.

| LossesSeverity class | Reported records | Approx. share |
|---|---:|---:|
| No-Loss | 49,545 | 75.785% |
| Seepage Loss | 12,880 | 19.701% |
| Partial Loss | 2,647 | 4.049% |
| Severe Loss | 270 | 0.413% |
| Complete Loss | 34 | 0.052% |
| **Total** | **65,376** | **100.000%** |

Severe plus complete loss accounts for only 304 records (about 0.465%). A classifier that always predicts No-Loss could score about 75.8% accuracy on this reported distribution while detecting none of the other classes. Accuracy must not be used alone.

## Data-audit gates before training

1. Load the original workbook in code and verify the row count, column names, data types, label spellings and class counts.
2. Confirm the units and definitions for every feature; specifically establish whether `HoleSection` is a hole diameter/section category and not a well identifier.
3. Verify that no feature leaks label information or is derived from the loss-severity classification process.
4. Check row order, depth ordering, duplicates, impossible values, repeated depth points, and whether rows can be mapped to one of the 20 wells.
5. Locate a reliable well identifier or obtain a documented row-to-well mapping. The 18-column list supplied so far does not include a clear well-ID field.

## Evaluation plan

### Reference experiment
Only after the raw data audit, run a simple, reproducible baseline with a fixed random seed and documented preprocessing. It can serve as a benchmark-style experiment, but row-level random splitting may place nearby depth records from the same well in both training and test sets. Do not present this as evidence of generalization to unseen wells.

### More demanding validation
Prefer a held-out-well test (grouped by well ID) so that entire wells are excluded from training. If well IDs cannot be recovered, record this limitation explicitly and do not claim out-of-well generalization.

### Baselines and metrics
- Majority-class dummy classifier, to establish a naïve accuracy baseline.
- A transparent classical model before deep learning (for example, class-weighted Random Forest or gradient boosting), with all preprocessing fitted on training data only.
- Confusion matrix, per-class precision/recall/F1, macro-F1, balanced accuracy, and class support.
- For any future alert/no-alert framing, define the alert threshold and report false alarms and missed events. Estimate lead time only if valid chronological data support it.
- Keep the rare Complete Loss class visible in all reports. If validation partitions contain too few complete-loss cases, mark the estimate as unstable rather than hiding the class or merging labels without justification.

## Experimental baseline added to the hosted prototype

The app now offers an optional Random Forest classifier compared with a majority-class dummy baseline after detecting `LossesSeverity`. It uses numeric predictor columns only, excludes the target from inputs, and runs a fixed-seed stratified 70/30 row-level split. It displays accuracy, balanced accuracy, macro-F1, per-class metrics, a confusion matrix, and model-specific feature importances.

**Critical limitation:** the visible columns do not include a clear well identifier. The row-level split may therefore put records from the same well or adjacent depths in both training and test partitions. These metrics are a reproducible exploratory benchmark only, not evidence of performance on unseen wells, other fields, or live operations. Class-5 support in the test split is very small and must be reported alongside the metrics. No model artifact is saved or promoted as operational.

## Additional depth-block sensitivity test

The hosted prototype also provides an optional *depth-block stress test*. It identifies the numeric `M.Depth` column, orders the distinct depth values, trains on records below the 70th-percentile distinct-depth cutoff, and tests on records at or above that cutoff. The same Random Forest, majority-class dummy baseline, and per-class metrics are reported for comparability.

This is a distribution-shift sensitivity check, **not** a well-held-out validation. The same physical well may contribute records to both partitions because the visible workbook has no clear well identifier. It can show whether scores degrade when tested at higher measured depths, but it cannot establish transfer to unseen wells. If a reliable well ID / row-to-well mapping is found, the preferred validation remains grouping the split by well.

## Training-only candidate comparison

The app now compares three candidate estimators using only an internal stratified validation subset from the outer training partition:

- Random Forest with `class_weight="balanced_subsample"` and `min_samples_leaf=2`.
- Random Forest with `class_weight="balanced"` and `min_samples_leaf=1`.
- Extra Trees with `class_weight="balanced"` and `min_samples_leaf=1`.

Candidates are ranked by validation Macro F1, with balanced accuracy as the tie-breaker. The selected candidate is then refit on the full outer training partition and evaluated on the existing held-out test partition. The held-out test labels are not used for candidate selection. The report retains each candidate's validation accuracy, balanced accuracy, Macro F1 and per-class metrics.

This is an exploratory selection policy, not exhaustive hyperparameter tuning. Macro F1 gives every class equal weight but may still be unstable when rare classes have very few validation cases. Severe/complete-loss recall and support must be examined separately. Candidate selection must not be repeated based on repeated inspection of held-out test metrics; if this happens, the test data become part of the tuning process and a new independent evaluation set is needed.

## Claims boundary

This is a single-field public research dataset. It cannot demonstrate performance on a different operator, formation, rig, basin, or data-acquisition system. A model with good offline scores would still be an experimental classifier—not a validated early-warning or operational drilling tool. Qualified drilling professionals and independent validation would be required before any field use.

## Candidate comparison update (2026-10-23)

The app now compares three candidate estimators using only an internal stratified validation subset carved from the outer training split: (1) Random Forest with `balanced_subsample` class weighting and leaf size 2; (2) Random Forest with `balanced` class weighting and leaf size 1; and (3) Extra Trees with `balanced` class weighting and leaf size 1. It chooses by the highest internal-validation Macro F1, breaking ties with balanced accuracy, then refits that estimator on all outer training records and evaluates it on the held-out test partition.

This reduces direct tuning on the held-out test set but does not eliminate all research limitations. The internal validation estimate for Complete Loss will be based on very few records, the depth-block experiment can still mix records from the same wells, and the selection rule does not replace domain review or well-held-out validation. Read validation Macro F1 as a model-selection aid, not a guarantee of reliability.

## Macro F1 scope clarification (2026-10-23)

The app now reports two explicit Macro F1 values:
- `macro_f1_all_known_classes`: macro F1 with the full set of known target classes passed explicitly. This strict aggregate assigns F1=0 to classes with no true test support and must be interpreted together with support counts.
- `macro_f1_test_present_classes`: macro F1 using only labels represented in the test labels or predictions.

Candidate selection on the internal validation split also passes the complete known class list explicitly to Macro F1. Balanced accuracy is still computed on classes represented in the true test labels; a class absent from the test set cannot have its recall measured. Always publish per-class support and recall alongside aggregate scores. If Complete Loss has zero test support, the experiment cannot validate Complete Loss detection, regardless of aggregate metrics.

