# HalaLoya AI Drilling Assistant — Research Protocol

## Current implementation status

The Streamlit prototype can inspect CSV/XLSX drilling datasets and, when the LossesSeverity column is detected, run optional offline experiments for (a) multiclass classification of the recorded severity and (b) a separate binary severe/complete-versus-other screen. Model fitting occurs during the app session. The app does not persist a trained model, connect to rig instruments, or create future-event forecasts.

The experiment is research-only. No operational performance claim is currently supported.

## Intended two-stage research objective

1. **Recorded-severity diagnosis:** classify the severity label attached to an observed row, then evaluate whether performance persists on independent groups.
2. **Early warning:** predict a future loss event or escalation within an engineer-defined horizon using chronological measurements available before that event. This needs different labels and data than same-row diagnosis.

## Data-governance gate

Before any dataset is used:
1. Record the source URL, owner, download date, version, and provenance.
2. Verify license terms and any third-party notices before research, modification, redistribution, publication, or commercial use.
3. Keep attribution and permission evidence with the research records.
4. Do not upload confidential company/well data to the public prototype without authorization and approved handling controls.
5. Preserve an untouched raw copy and document every cleaning/transformation step.

Publicly downloadable does not automatically mean freely redistributable or commercially usable.

## Minimum requirements for recorded-severity diagnosis

- Recompute row count, column names, data types, missingness, duplicates, label values, and class counts directly from the exact workbook being tested.
- Maintain a data dictionary for every feature, including engineering meaning, units, range, source, and unresolved questions.
- Verify the target encoding; in particular, test both integer and floating-point encodings and do not silently treat all rows as non-critical if labels fail to parse.
- Exclude the target and measured depth from the current experimental predictors. The depth-block split may use depth only to determine which rows are held out.
- Treat reset-based row segments as **proxy groups**, not confirmed wells. Confirm the row-to-well mapping with source records or the dataset authors before claiming an unseen-well evaluation.
- Do not treat combinations of similar mud-property values as true report days or wells. Such signatures may mix observations from different wells and can create a very large number of groups.
- Compare to simple baselines and report each class's support, precision, recall, F1, confusion matrix, balanced accuracy and macro-F1. For severe/complete-loss screening, report average precision relative to prevalence and recall within a fixed alert budget.
- Fit/choose preprocessing and models only within the training groups. Do not repeatedly tune thresholds or models on a final holdout and continue to call it independent.
- Include uncertainty estimates, but state when they are based on very few independent wells or positive-event groups.

## Minimum requirements for early warning

The current Azadegan workbook is described as depth-based, meter-by-meter data built from drilling, mud-logging, daily and final well reports. The published field description does not establish a timestamp and real Well ID in the 18-column workbook schema used here. Therefore, this workbook alone cannot establish a prospective “loss in the next 30 minutes” target without additional timing and event records.

Required data include:
- A real well identifier and timestamp for each measurement, with documented sampling intervals.
- Accurate event onset and end times, onset severity/volume, and treatment/intervention logs.
- Operating actions and context that distinguish changes caused by a loss from changes made by the drilling team.
- A target horizon agreed with drilling engineers and based on response time and sampling cadence.
- Time-ordered features made only from information available before the prediction time.
- Validation grouped by well and ordered in time, reporting detected critical events, lead time, missed events, and false alarms per drilling hour.

## Known dataset risks

- The published Azadegan data represent 20 wells and five highly imbalanced classes; only 270 Severe Loss and 34 Complete Loss records are reported.
- The source table describes Flow In in gallons per minute (gpm) and Flow Out as a percentage (0–100). They cannot be subtracted directly as if they shared a unit.
- The visible feature list has no clear Well ID or timestamp. HoleSection is a hole-section/diameter field, not a well identifier.
- A depth-based split may exclude some classes completely. Zero support for a class means recall for that class cannot be evaluated on that split.

## Prohibited claims at this stage

Do not claim validated prediction accuracy on unseen wells/fields, prospective early-warning skill, improved safety, real-time performance, or operational readiness. Current metrics are offline exploratory results only.

## Safety boundary

Do not connect the prototype to rig controls or use it to make real-time drilling, well-control, treatment, or other safety-critical decisions. Any future field trial must be reviewed by qualified drilling professionals and run in shadow mode before any workflow relies on it.


## Public deployment smoke test (2026-10-25)

A read-only browser smoke test was performed on https://ehvqysekxk9hq7maczez3x.streamlit.app/ .

Observed:
- The page loaded successfully and rendered the title "HalaLoya AI Drilling Assistant".
- The upload control for CSV/XLSX was visible.
- The public page displayed a clear research-only warning stating that the prototype is not a real-time early-warning system, is not validated for drilling operations, and must not be used for safety-critical decisions.
- The page warned against uploading confidential or commercially sensitive company data.
- No page-level error or exception trace was observed during this opening-page check.

Scope limits:
- This was a page-load and visible-interface smoke test only. No workbook was uploaded and no model experiment was executed in the browser during this check.
- The test does not prove that the live deployment is running the exact latest GitHub commit, nor does it verify model results or data-download behavior.
- Before claiming deployment-level model validation, run a controlled end-to-end test with an authorized non-confidential copy of the public research workbook, record the app's displayed label counts and metrics, and compare them with the reproducible local evaluation artifacts. Do not upload confidential or commercially sensitive data.

