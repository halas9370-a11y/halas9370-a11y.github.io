# HalaLoya AI Drilling Assistant — Research Protocol

## Current status
This repository contains an exploratory CSV inspection prototype. It does not train a machine-learning model, predict lost circulation, or provide operational recommendations.

## Data governance gate
Before using any dataset:
1. Record the original source URL, owner, download date, and dataset version.
2. Verify the license or obtain written permission for research, modification, redistribution, and any commercial use that may apply.
3. Preserve required attribution and notices.
4. Do not upload confidential company or well data without authorization.
5. Keep a copy of the license/permission evidence with the research records.

Publicly downloadable does not automatically mean freely reusable or redistributable.

## Minimum requirements before a lost-circulation model
- A clearly defined event label and documented labeling method.
- Documented well identifiers and a review for leakage between training and test sets.
- Data dictionary, units, sampling interval, missingness analysis, and provenance.
- A baseline model before trying complex methods.
- Evaluation split by well (not random rows from the same well across both sets).
- Metrics appropriate to rare events, including precision, recall, false-alarm rate, and lead time where timestamps support it.
- Confidence/uncertainty reporting and review by qualified drilling professionals.
- Independent validation before any real-world use.

## Prohibited claims at this stage
Do not claim prediction accuracy, early-warning capability, improved safety, or operational readiness. No model has been trained or validated by this prototype.

## Safety boundary
This software is for exploratory research and education only. It must not be connected to rig controls or used to make real-time drilling, well-control, or safety decisions.
