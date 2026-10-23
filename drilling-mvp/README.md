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
python halaloya_grouped_eval.py /path/to/42452_2021_4769_MOESM1_ESM.xlsx --bootstrap-reps 50 --out-json grouped_eval.json
```

The quick default uses 50 group-bootstrap resamples; use `--bootstrap-reps 300` or more if runtime allows and tighter uncertainty summaries are needed. The script checks row order and reconstructs **proxy segments**, not real wells. Its grouped scores must not be reported as leave-one-well-out validation unless the row-to-well mapping is verified. Results describe detection of same-row recorded labels, not a future-event early-warning model.

## Source

Mardanirad, Wood & Zakeri (2021), “The application of deep learning algorithms to classify subsurface drilling lost circulation severity in large oil field datasets,” *SN Applied Sciences*. https://doi.org/10.1007/s42452-021-04769-0

This repository is an exploratory research artifact, not a validated prediction system. Independent validation and review by qualified drilling professionals are prerequisites before any field use.
