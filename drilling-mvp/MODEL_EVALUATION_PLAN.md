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

## Grouped-proxy audit v2 on the original Azadegan workbook (2026-10-25)

### What was executed

The supplied workbook was read directly from `42452_2021_4769_MOESM1_ESM.xlsx` and checked by a hardened script, with an expected critical-row count guard of 304. The binary label parser was also tested on `Severe_Loss`, repeated spaces, numeric 4.0/5.0, and strings "4.0"/"5". All are recognized by the current parser. Therefore, a claim that these specific cases silently become zero is not true for the current repository version; it may describe an older copy or a different implementation.

The run used `HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=30, class_weight="balanced", early_stopping=False)`, leave-one-inferred-segment-out scoring, and 50 group-bootstrap repetitions. The 30-iteration setting is a **quick exploratory audit**, not a chosen final model. All metrics remain retrospective same-row screening results.

### Data and row-order findings

- 65,376 rows, 18 columns, zero missing cells, zero exact duplicate rows.
- Recorded class counts: No-Loss 49,545; Seepage Loss 12,880; Partial Loss 2,647; Severe Loss 270; Complete Loss 34. Severe + Complete = 304 rows (0.465%).
- A model that always predicts noncritical has 99.535% accuracy but zero critical recall. Accuracy alone is therefore uninformative for this task.
- At a 50 m reset threshold, row order yields 22 inferred segments, of which only 5 contain any critical row and 2 contain Complete Loss. Seven inferred segments start deeper than 500 m.
- Sensitivity to reset threshold: 23 segments at 10 m; 22 at 50 m; 21 at 200 m; 20 at 500 m. Matching the published count at 500 m is not proof of correct well identity.
- Of 21 boundaries at 50 m, 18 also change HoleSection and all 21 show a change in at least one mud-property field. These are supporting clues, not a validated row-to-well map.
- The shuffled-depth negative control yields 31,863 segments at 50 m, versus 22 in the original ordering; only about 0.26% of shuffled adjacent steps are within 5 m, compared with 99.95% in the original. This strongly supports that the workbook is depth-ordered. It does **not** prove every inferred segment is a complete physical well.
- A sampled nearest-neighbour audit (1,000 rows, features exclude target and depth) found median nearest-neighbour distance 0.866 within the same proxy segment versus 1.620 across proxy segments; in 13.1% of paired sampled rows the nearest cross-segment neighbour was at least as close as the nearest same-segment neighbour. This flags some feature-space overlap but cannot prove leakage or physical well identity.
- A one-vs-rest adversarial distribution-separability audit for each segment versus all other segments gave median ROC-AUC 0.943 (range 0.833–0.995). High AUC means the segment's feature distribution is distinguishable from other segments; low AUC would mean it is not distinguishable under this setup. Neither high nor low AUC establishes whether segments correspond to different wells. This AUC is a distribution-shift diagnostic, not an identity test.
- The model run remains leave-one-**proxy-segment**-out, not leave-one-well-out. A real Well ID or documented row-to-well map is still required.

### Revised metric methodology

The previous procedure compared a bootstrap interval for average precision (AP) with a single prevalence value from the full dataset. That comparison does not account for prevalence varying between bootstrap samples. It also silently omitted bootstrap samples without positive groups and pooled probability scores from different fitted fold models.

The revised script now:
- explicitly guards the recognized critical count when `--expected-critical-count 304` is supplied;
- disables automatic early stopping and records a fixed iteration count;
- reports the number of bootstrap replicates with no positive groups that are excluded from the undefined AP-delta summary;
- reports per-segment AP and AP minus that segment's local prevalence, then bootstraps segments;
- ranks rows **within each held-out segment** for alert-budget metrics rather than treating probabilities from different fold-specific models as directly comparable;
- reports alert precision/row recall and a proxy-run detection metric; and
- labels pooled out-of-fold AP as secondary/uncalibrated across fold models.

The "proxy runs" are simply contiguous Severe/Complete-labelled rows within an inferred segment. With no event timestamp or event ID, they are not confirmed physical loss events.

### Quick grouped-proxy results (30 iterations; 50 bootstrap repetitions)

Critical prevalence in the data is 0.465%. The values below are exploratory and conditional on the inferred groups and fixed out-of-segment scores.

| Feature set | Pooled OOF AP (secondary only) | Macro AP within the 5 positive proxy segments | Bootstrap median of mean (segment AP − local prevalence), 95% interval | Row recall at top 5% alerts per segment (median, 95% interval) | Proxy-run recall at top 5% alerts per segment (median, 95% interval) |
|---|---:|---:|---:|---:|---:|
| All numeric features except target and M.Depth | 0.0476 | 0.1232 | 0.0941 (0.0224–0.2750) | 0.194 (0.000–0.396) | 0.316 (0.000–1.000) |
| Without FlowIn + FlowOut + PumpStroke | 0.0273 | 0.1158 | 0.0986 (0.0097–0.2878) | 0.324 (0.005–0.627) | 0.224 (0.025–0.500) |
| Without that family and without StandpipePressure | 0.0258 | 0.1591 | 0.1353 (0.0128–0.3325) | 0.296 (0.019–0.531) | 0.239 (0.009–0.731) |
| FlowOut-only directional ranking heuristic (not an alarm rule) | 0.0107 | 0.2353 | 0.2144 (0.0940–0.3748) | 0.336 (0.095–0.532) | 0.703 (0.500–0.810) |

For the first feature set, 1 of 50 bootstrap replicates contained no positive groups and was excluded from the AP-delta statistic; that count is now visible. All intervals are based on only 22 proxy segments, with positives in five of them. They do not incorporate full model retraining uncertainty. The event-like metric uses 13 contiguous positive-label runs, not independently confirmed events or timestamps.

### Interpretation and remaining blockers

- The ranking signals vary by feature set, and some segment-level AP differences exceed local prevalence. However, the number of positive proxy segments is small, the grouping is not verified, and the model run uses only 30 fixed boosting iterations. These results are not enough to claim a reliable detector.
- The FlowOut-only heuristic is a retrospective comparator. It must not be promoted to an alarm rule without validating sensor meaning, operating context and engineering costs. FlowIn is reported in gpm and FlowOut in percent; subtracting these units directly is invalid.
- The grouped script still performs a **binary** Severe/Complete-versus-other evaluation. It is not a substitute for the separate Phase 1 multiclass evaluation across all five recorded severity classes.
- AUC near 0.5 in adversarial validation does not prove two groups are the same well; it only indicates that the chosen feature distribution was not distinguishable in this test.
- Early warning remains out of scope until real timestamps, Well IDs, event-onset times and intervention records are available.

### Reproducible command

```bash
python halaloya_grouped_eval.py 42452_2021_4769_MOESM1_ESM.xlsx --allow-depth-reset-proxy --expected-critical-count 304 --bootstrap-reps 50 --max-iter 30 --out-json HalaLoya_Azadegan_grouped_eval_v2.json --segments-csv HalaLoya_Azadegan_segments_v2.csv
```

Before publication, rerun with a larger bootstrap count (e.g. 300 or more) and pre-specified fixed iteration settings, and do not describe these proxy groups as wells without a source mapping.
