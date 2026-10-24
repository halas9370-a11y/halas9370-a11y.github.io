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

## Phase 1 — five-class diagnosis on inferred depth-reset proxy groups (2026-10-25)

### Experiment definition

The original Azadegan workbook was evaluated with a separate five-class experiment. Each row received an out-of-group prediction from a model trained without its inferred depth-reset segment. Group boundaries were inferred whenever measured depth fell by more than 50 m between adjacent rows. These are **proxy segments, not verified well IDs**.

- Rows: 65,376; columns: 18; missing cells: 0; exact duplicate rows: 0.
- Label counts: No-Loss 49,545; Seepage Loss 12,880; Partial Loss 2,647; Severe Loss 270; Complete Loss 34.
- Severe + Complete: 304 rows (0.465%).
- Predictors: 16 numeric columns, excluding both LossesSeverity and M.Depth.
- Model: HistGradientBoostingClassifier, max_depth=4, learning_rate=0.05, max_iter=30, class_weight=balanced, early_stopping=False, random_state=0.
- Validation: leave-one-inferred-depth-reset-proxy-segment-out across 22 segments. No verified Well IDs are available in the workbook.

### Main results

| Metric | Model | Per-fold training-majority baseline |
|---|---:|---:|
| Accuracy | 67.12% | 75.78% |
| Balanced accuracy | 29.56% | 20.00% |
| Macro F1 (all five classes) | 0.281 | 0.172 |
| Weighted F1 | 0.693 | 0.653 |
| Quadratic weighted kappa | 0.282 | 0.000 |
| Ordinal MAE (class distance) | 0.405 | 0.292 |

The model gives up overall accuracy compared with the majority baseline, while improving balanced accuracy and macro F1. That is a trade-off, not a general win: the critical classes still perform very poorly. The ordinal MAE is worse than the baseline, so the model is not consistently improving distance-sensitive severity errors.

### Per-class results

| Recorded class | Precision | Recall | F1 | True test support | Model predictions |
|---|---:|---:|---:|---:|---:|
| No-Loss | 84.73% | 74.11% | 0.791 | 49,545 | 43,334 |
| Seepage Loss | 38.50% | 51.61% | 0.441 | 12,880 | 17,267 |
| Partial Loss | 14.11% | 19.12% | 0.162 | 2,647 | 3,586 |
| Severe Loss | 0.73% | 2.96% | 0.012 | 270 | 1,095 |
| Complete Loss | 0.00% | 0.00% | 0.000 | 34 | 94 |

The model predicted Complete Loss 94 times but got zero Complete Loss cases correct. Of the 34 actual Complete Loss rows, 26 were predicted as Severe Loss, 7 as No-Loss and 1 as Seepage Loss. Of 270 actual Severe Loss rows, only 8 were predicted correctly as Severe Loss. These results do not support a critical-loss detector claim.

### Proxy-group bootstrap uncertainty

A 1,000-repetition bootstrap resampled the 22 inferred proxy segments and recomputed metrics from fixed out-of-segment predictions. The 95% intervals are:

| Metric | Bootstrap median | 95% interval |
|---|---:|---:|
| Accuracy | 0.673 | 0.578–0.756 |
| Balanced accuracy | 0.300 | 0.236–0.479 |
| Macro F1 | 0.278 | 0.224–0.346 |
| Weighted F1 | 0.695 | 0.592–0.776 |
| Quadratic weighted kappa | 0.283 | 0.130–0.421 |
| Ordinal MAE | 0.402 | 0.291–0.537 |

Complete Loss was absent from 120 of 1,000 bootstrap samples, because the class appears in only two inferred proxy segments. In the 880 samples that contained Complete Loss support, recall remained exactly zero. These intervals condition on the same proxy grouping and fixed predictions; they do not account for full model-training uncertainty or uncertainty in the physical identity of the groups.

### Quick feature-family sensitivity test

A separate ablation used the same proxy grouping but a different, deliberately lighter model configuration (max_iter=10, max_bins=63, min_samples_leaf=30). Therefore, do not compare its absolute metrics directly with the 30-iteration main model. Compare only the variants within this quick ablation.

| Feature set | Accuracy | Balanced accuracy | Macro F1 | Severe recall | Complete recall |
|---|---:|---:|---:|---:|---:|
| All numeric features except target and depth | 62.94% | 29.86% | 0.276 | 3.33% | 0.00% |
| Without FlowIn, FlowOut and PumpStroke | 60.16% | 28.79% | 0.263 | 4.07% | 0.00% |
| Without flow family and StandpipePressure | 57.20% | 28.85% | 0.252 | 4.44% | 0.00% |

Removing the flow family modestly reduced aggregate scores while slightly increasing Severe recall; Complete recall remained zero in every variant. This is not enough to conclude that the flow features leak the target, or that no leakage exists. The label derivation process must be documented, and related features can carry similar physical information.

### Decision and next steps

1. Do not tune the current model further against these same proxy folds and then treat them as an independent test.
2. Obtain actual Well IDs or a source-authenticated row-to-well mapping. The 22 inferred segments cannot be labelled independent wells.
3. Keep the multiclass evaluation and the binary Severe/Complete screen as separate tasks; neither evaluates future event onset.
4. Prioritize analysis of why Complete Loss cases are mapped to Severe Loss and why Severe Loss has recall under 3% in the main run. Candidate changes must be selected using group-respecting validation, followed by a new independent holdout if available.
5. Seek timestamps, event-onset labels and intervention logs before developing the second-stage early-warning model.

Reproducible local script: `drilling-mvp/halaloya_multiclass_proxy_eval.py`. The run must explicitly include `--allow-depth-reset-proxy` and should use `--expected-critical-count 304` for the supplied workbook.

### Flow-family ablation and critical-union view

A separate quick sensitivity run used 10 boosting iterations, 63 bins, and minimum leaf size 30. This is a different model configuration from the main 30-iteration run, so compare feature variants only **within this table**:

| Features | Accuracy | Balanced accuracy | Macro F1 | Severe recall | Complete recall |
|---|---:|---:|---:|---:|---:|
| All numeric features except target/depth | 62.94% | 29.86% | 0.276 | 3.33% | 0.00% |
| Remove FlowIn, FlowOut, PumpStroke | 60.16% | 28.79% | 0.263 | 4.07% | 0.00% |
| Also remove StandpipePressure | 57.20% | 28.85% | 0.252 | 4.44% | 0.00% |

Removing flow-related inputs reduced aggregate metrics slightly and increased Severe recall by a small amount, while Complete recall remained zero. This is a sensitivity result, not evidence that flow variables leak the target or are uninformative.

As a secondary calculation from the main five-class model's predictions, treating either predicted Severe or predicted Complete as a binary critical alert yields 40 true positives out of 304 actual critical rows (13.16% recall), 1,149 false positives, and 3.36% precision. This is **not** the separately trained binary critical-loss model; it only aggregates the five-class model's predictions. It confirms that the multiclass model is not a useful critical-loss alert at this stage.

A full reproduction of the multiclass run, including out-of-proxy-segment predictions and confusion matrix, is available in `drilling-mvp/halaloya_multiclass_proxy_eval.py`. The script requires explicit `--allow-depth-reset-proxy` opt-in and supports a critical-row count guard.

### Grouping-threshold sensitivity (quick five-class check)

To test whether the apparent multiclass result depends only on the chosen reset threshold, the same quick configuration was rerun with 10, 50, 200 and 500 m thresholds (10 iterations, 63 bins, minimum leaf size 30). These are within-experiment comparisons only; do not compare them directly to the primary 30-iteration model.

| Reset threshold | Inferred segments | Accuracy | Balanced accuracy | Macro F1 | Severe recall | Complete recall |
|---:|---:|---:|---:|---:|---:|---:|
| 10 m | 23 | 62.85% | 29.84% | 0.276 | 3.33% | 0.00% |
| 50 m | 22 | 62.94% | 29.86% | 0.276 | 3.33% | 0.00% |
| 200 m | 21 | 63.08% | 29.99% | 0.277 | 3.33% | 0.00% |
| 500 m | 20 | 62.49% | 29.83% | 0.275 | 3.33% | 0.00% |

Scores are similar across these proxy-group thresholds, and Complete Loss recall remains zero. Changing the reset threshold alone does not solve the rare-class diagnosis problem. The fact that the 500 m threshold produces 20 segments is **not proof** those segments correspond to the 20 physical wells.
