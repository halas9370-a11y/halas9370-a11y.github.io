# HalaLoya AI Drilling Assistant — Initial Model Evaluation Plan

## Purpose and status

This document records a cautious plan for a research-only first supervised-learning experiment using the publicly published Azadegan oilfield lost-circulation dataset. The deployed-app source in GitHub supports optional offline model experiments when `LossesSeverity` is detected. It can compare candidate classifiers for recorded-label diagnosis and run a separate same-row Severe/Complete-versus-other screen. The models are fitted within the app session and are not saved as reusable or operational models. The live deployment could not yet be confirmed in this review.

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

## Depth-band label distribution audit

The app now groups the dataset into up to five quantile-based measured-depth bands and reports both the number and within-band percentage of each observed `LossesSeverity` class. This audit helps test whether classes are concentrated in certain depth ranges and whether a depth-block test excludes rare classes. It is descriptive only: it does not establish that depth causes loss severity, does not validate predictions, and cannot replace well-level split validation.

## Observed depth-band pattern from the supplied dataset (2026-10-23)

The current 65,376-row Azadegan workbook shows clear changes in observed label mix across measured-depth bands. Based on the provided count table:

| Measured-depth band | No-Loss | Seepage Loss | Partial Loss | Severe Loss | Complete Loss |
|---|---:|---:|---:|---:|---:|
| 14–985 | 97.62% | 2.25% | 0.14% | 0.00% | 0.00% |
| 985–1,739 | 84.04% | 10.34% | 4.22% | 1.15% | 0.25% |
| 1,739–2,432 | 74.10% | 19.52% | 6.22% | 0.16% | 0.00% |
| 2,432–3,079 | 63.09% | 32.14% | 4.76% | 0.01% | 0.01% |
| 3,079–4,285 | 60.04% | 34.30% | 4.91% | 0.75% | 0.00% |

These counts show that the depth-block test at `M.Depth = 3004` has zero Complete Loss support in its higher-depth test block: all 34 Complete Loss records fall below the threshold. This split therefore cannot evaluate Complete Loss recall. The 98 Severe Loss examples in the deepest band are all part of the higher-depth test block in the reported audit, and the current candidate did not predict any Severe Loss examples there.

Interpretation: the depth-block test indicates a meaningful shift in the label mix and a serious failure to identify severe losses in the deeper block. It does **not** establish why this shift occurs or prove that depth causes fluid loss. Factors may include different wells, formations, operational contexts, or a combination of variables. Do not tune repeatedly against the same held-out block and then present it as an independent test.

## External dataset candidate: Marun oil field

A separate public dataset is described for the Marun oil field (southwest Iran): 2,820 records extracted from drilling, mud, and final well reports for 305 wells. The published study describes 18 input variables and one continuous output, mud-loss severity rate (bbl/hr), rather than the same five-class `LossesSeverity` label in the Azadegan workbook. Source article: https://doi.org/10.1016/j.petrol.2020.108125. Dataset landing page: https://data.mendeley.com/datasets/hr7hbxdryp/1

The Mendeley Data listing states a CC BY-NC 3.0 license. Treat this as **non-commercial research only** unless separate permission is obtained. Do not package, redistribute, or use it commercially in a paid service under this license without verifying rights and obtaining any required permission. This dataset is not a drop-in external test for the current classifier: its target is continuous, feature schema and units differ, and a Complete Loss class cannot be inferred safely from loss rate alone without a documented label for total loss of returns. It may support a separate regression-research track after checking the workbook schema, units, labels, provenance, and allowed use.

## Next validation priorities

1. Stop using higher-depth accuracy as the only criterion for improvement; report the complete absence of Complete Loss examples in this holdout every time.
2. Determine whether row-to-well identity or a reliable mapping can be obtained from the authors or an original 22-variable database. The published cleaned workbook used here does not expose an obvious well ID in its 18 columns; `HoleSection` must not be treated as a well identifier.
3. Keep candidate tuning restricted to the training/validation data. Once model selection is complete, evaluate exactly once on a properly defined final test.
4. In parallel, inspect the Marun workbook only as a separately licensed research candidate. Do not claim direct cross-field validation until targets and features are made genuinely comparable.

## Additional candidate estimators (2026-10-23)

The optional training-only model comparison now includes five estimators:
1. Random Forest with `balanced_subsample` weights and `min_samples_leaf=2`.
2. Random Forest with `balanced` weights and `min_samples_leaf=1`.
3. Extra Trees with `balanced` weights and `min_samples_leaf=1`.
4. Decision Tree with `class_weight="balanced"` and `min_samples_leaf=2`.
5. AdaBoost with a shallow decision-tree base estimator and sample weights computed with balanced class weights.

The inclusion of Decision Tree and AdaBoost is motivated by the 2022 published analysis of the same Azadegan dataset, which reports strong class-specific results from tree-based methods and notes that Decision Tree performs well for complete loss while AdaBoost performs well for severe loss. This is motivation to test candidates, not a guarantee they will work under the current depth-block split. Source: https://doi.org/10.1007/s13202-021-01411-y

The candidates are selected by internal-validation Macro F1 over all known classes, with balanced accuracy as a tie-breaker. The chosen estimator is refit on the outer training partition before evaluation on the held-out partition. This remains exploratory: depth-block holdout is not well-held-out validation, and the rare complete-loss class is absent in the high-depth block. The selected-model test accuracy is now labelled generically rather than as Random Forest accuracy.

## Separate binary critical-loss screen (2026-10-23)

Because the multiclass candidate comparison selected models that predicted zero Severe Loss and Complete Loss in the high-depth test block, the app now also runs a distinct binary screening experiment: `Severe Loss` or `Complete Loss` versus all other recorded classes.

Three class-weighted candidates (Random Forest, Extra Trees, and Decision Tree) are compared using only the same inner validation subset carved from the outer training rows. For each candidate, a probability threshold is selected to maximize positive-class F2, with recall, precision, and then the higher threshold as tie-breakers. The selected candidate is refit on all outer training rows, and its binary metrics are evaluated on the held-out depth block. The report includes precision, recall, F1, F2, specificity, false positives, false negatives, true positives, and the chosen threshold.

This is a screening experiment on recorded labels, not a future-event early-warning model. The depth-block test has no Complete Loss records, so it cannot evaluate Complete Loss independently. A higher recall threshold may substantially increase false alarms. Any metric changes must be interpreted together with missed critical cases and false alarms, and this experiment does not establish transfer to unseen wells or operational safety.

## Diagnostic metrics for zero critical-loss detections (2026-10-23)

The binary critical-loss screen now reports the threshold selected using internal validation, average precision on the held-out depth block, and the maximum/median predicted critical-class probability for actual critical and non-critical test records. These metrics help determine whether the chosen threshold is above all test scores or whether the model poorly ranks deeper severe-loss cases.

The held-out test probabilities are diagnostic only. Do not use test results to choose a replacement threshold and then report those same results as independent evaluation. If the test scores are low for actual severe-loss records, prioritize the data representation and validation design (well identity, neighboring-depth/context variables, chronology, formation/operational metadata) rather than threshold changes alone.

## Foundation cleanup and grouped-screen safeguards (2026-10-24)

- The app code excludes M.Depth from numeric predictor features for recorded-label experiments. Depth remains available for descriptive depth-band analyses and to define the depth-block holdout only. The evaluation schema version is incremented when predictor/evaluation logic changes so stale session metrics are discarded.
- The original Azadegan workbook was subsequently tested with a grouped-proxy evaluation. Results and uncertainty summaries below are exploratory; the inferred groups are **not verified well IDs**.
- The target labels in the uploaded workbook are text values (No-Loss, Seepage Loss, Partial Loss, Severe Loss, Complete Loss), so the numeric 4.0/5.0 issue is not present in this file. The hardened script still handles text and numeric target encodings in case other copies differ.
- Depth-reset segments are only proxy groups. Do not equate similar mud-property signatures to calendar days or independent wells. The actual file produces 22 segments, compared with the 20 wells described in the source paper; seven inferred segments start deeper than 500 m, including a short segment covering only 3,704–4,285 m. The row-to-well mapping must be obtained or confirmed before claiming leave-one-well-out validation.
- Published Table 2 describes FlowIn in gpm and FlowOut in percent (0–100, with reported values up to about 100.1). Do not subtract these fields directly. A pressure/flow rule must be defined and reviewed with an engineer after verifying FlowOut sensor semantics, calibration, and operating context.
- Average precision should be compared with positive-class prevalence, and recall at a fixed alert budget should be reported alongside group-bootstrap intervals. The intervals below resample the inferred proxy groups while holding out-of-group scores fixed; they do not capture full model-fitting uncertainty.

## Observed grouped-proxy critical-loss screen on Azadegan (2026-10-24)

### Dataset audit

- 65,376 rows and 18 columns; no missing cells or exact duplicate rows detected.
- Class counts: No-Loss 49,545; Seepage Loss 12,880; Partial Loss 2,647; Severe Loss 270; Complete Loss 34.
- Severe plus Complete Loss: 304 rows (0.465%).
- An always-noncritical classifier has about 99.535% accuracy and zero critical recall. This demonstrates why accuracy is not a useful standalone metric for this rare-event task.

### Proxy-group reconstruction

- 99.966% of adjacent depth steps are non-decreasing and 99.954% of absolute steps are 5 m or less, but the reset heuristic creates 22 segments rather than the 20 wells documented in the paper.
- Seven of the 22 inferred segments start deeper than 500 m. Only five inferred segments contain any Severe/Complete rows; Complete Loss rows occur in only two proxy segments.
- Therefore, the run is leave-one-**proxy-segment**-out, not leave-one-well-out. These grouped results diagnose sensitivity to a plausible row-order grouping, but cannot establish performance on an unseen physical well.

### Results

The exploratory model is a HistGradientBoostingClassifier (max_depth=4, learning_rate=0.05, max_iter=200, balanced class weights) using numeric predictors while excluding M.Depth. It was evaluated with out-of-proxy-segment predictions. The positive prevalence baseline is 0.00465.

| Experiment | Full out-of-group AP | Group-bootstrap AP median (95% interval) | Recall in top 5% alert budget (median; 95% interval) |
|---|---:|---:|---:|
| Numeric features including FlowIn and FlowOut | 0.0373 | 0.0401 (0.0010–0.1094) | 0.421 (0.000–0.857) |
| Same features without FlowIn and FlowOut | 0.0315 | 0.0345 (0.0017–0.0857) | 0.332 (0.004–0.756) |
| Simple ranking heuristic: lower FlowOut% ranks higher concern | 0.0107 | 0.0123 (0.0041–0.0503) | 0.167 (0.002–0.500) |

The group-bootstrap summary used 300 replicates for this report. The AP intervals for both multivariable variants include or fall below the prevalence baseline, while intervals are very wide. The apparent lift in median AP is not sufficient evidence of a reliable detector because only five proxy groups contain critical examples and the groups are not verified wells. The FlowOut-only heuristic is a directional benchmark, not an approved alert rule. The with-flow / without-flow comparison does not prove or disprove target leakage.

### Interpretation boundary

This is retrospective same-row classification of recorded Severe/Complete labels. It is **not** a prediction of loss onset in the next 30 minutes or any other future horizon. A prospective early-warning model requires real Well IDs, timestamps, accurate loss-onset times and intervention logs, an engineer-defined horizon, and testing on future periods and entirely held-out wells. Do not use these findings for live drilling, well-control, or safety decisions.
