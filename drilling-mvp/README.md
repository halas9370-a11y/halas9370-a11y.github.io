# HalaLoya AI Drilling Assistant

A research prototype for exploring drilling-data quality and conducting **offline, exploratory** lost-circulation severity experiments.

## Current capabilities

- Upload CSV or Excel (.xlsx) data and select a worksheet.
- Inspect column types, missing cells, duplicate rows, numeric summaries, and simple charts.
- Inspect observed LossesSeverity class counts and how the recorded classes vary across measured-depth bands.
- Run optional experimental classification after selecting the model-comparison button.
- Compare candidate tree-based models on an internal validation split, then evaluate the selected candidate on a separate row-level or depth-block test partition.
- Run a separate severe/complete-loss screen for the labels recorded on the same row.
- Download a JSON evaluation report.

The app fits models in memory during the selected experiment. It does **not** save a reusable trained model, ingest live rig data, or forecast the time of a future loss event.

## Important validation boundaries

- The current exploratory model comparison excludes both the target column and M.Depth from predictors. Depth can be used to define the depth-block test or produce descriptive depth summaries, but is not a predictive input in this experiment.
- A random row split can put records from the same well or neighbouring depths on both sides. A depth-block split is a stress test, **not** a substitute for holding out complete, identified wells.
- The workbook's visible columns do not identify a reliable well ID or timestamp. Reconstructed row-order segments are hypotheses until checked against a real well mapping.
- The label classes are severely imbalanced. Review per-class support, precision/recall/F1, confusion matrices, and critical-event misses—not accuracy alone.
- The binary severe/complete-loss screen classifies the loss label recorded for a row; it is not a future-event early-warning model.
- A public deployment is not an approved place for confidential operator or well data. Upload only data you are authorized to share, and do not use this prototype for live drilling or safety decisions.

## Research files

- app.py — Streamlit prototype and optional exploratory model experiments.
- MODEL_EVALUATION_PLAN.md — evaluation rationale and known limitations.
- RESEARCH_PROTOCOL.md — claim, validation, and data-governance boundaries.
- DATASET_CANDIDATE.md — Azadegan source and reuse notes.
- DATA_DICTIONARY.md — published variable meanings, units, and unresolved checks.
- halaloya_grouped_eval.py — grouped proxy-segment evaluation for severe/complete loss screening.

## Grouped critical-loss screening (local/offline)

Run from the drilling-mvp directory:

```bash
python halaloya_grouped_eval.py /path/to/42452_2021_4769_MOESM1_ESM.xlsx --allow-depth-reset-proxy --expected-critical-count 304 --bootstrap-reps 50 --max-iter 30 --out-json grouped_eval.json --segments-csv inferred_segments.csv
```

The `--allow-depth-reset-proxy` option is required because the inferred groups are not verified Well IDs; the command runs a **proxy-segment** analysis only. The expected-count guard checks that the text label parser recognizes all 304 Severe/Complete rows in the supplied workbook, and stops before fitting if it does not. `--max-iter 30` and `--bootstrap-reps 50` produce a quick exploratory audit, not a final model selection. The script disables automatic early stopping, reports threshold-sensitivity and shuffled-order diagnostics, nearest-neighbour overlap and adversarial segment-distribution AUC, within-segment ranking metrics, bootstrap exclusions, and contiguous positive-label runs as proxy episodes. Those runs are not confirmed physical events.

## Source

Mardanirad, Wood & Zakeri (2021), “The application of deep learning algorithms to classify subsurface drilling lost circulation severity in large oil field datasets,” *SN Applied Sciences*. https://doi.org/10.1007/s42452-021-04769-0

This repository is an exploratory research artifact, not a validated prediction system. Independent validation and review by qualified drilling professionals are prerequisites before any field use.
