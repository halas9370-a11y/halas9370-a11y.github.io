# HalaLoya AI Drilling Assistant — Initial Data Dictionary

## Dataset provenance

The Azadegan supplementary workbook is associated with Mardanirad, Wood & Zakeri (2021), *The application of deep learning algorithms to classify subsurface drilling lost circulation severity in large oil field datasets*. The article describes 65,376 depth-based records from 20 wells and reports 17 influencing variables plus the recorded loss-severity target.

Source: https://doi.org/10.1007/s42452-021-04769-0

**Important:** the units and ranges below are transcribed from the published article's Table 2. Before modelling, verify the raw workbook's units, numerical scale, column names, and label encoding directly. The paper describes preprocessing/normalization for its own model; do not assume every distributed copy is raw physical-unit data without checking observed values.

## Features

| Workbook column | Meaning | Published unit | Published range | Notes / checks |
|---|---|---|---|---|
| HoleSection | Hole section / nominal hole diameter | inch | 26–4.125 | Not a well ID. Check whether the raw file stores sizes as numbers or categories. |
| M.Depth | Measured depth | m | 14–4,285 | Used for depth audits and the optional depth-block split; excluded from predictors in the current app evaluation. |
| RateofPenetration | Rate of penetration (ROP) | m/hr | <1–616 | Check values and sensor/report context. |
| WeightonBit | Weight on bit (WOB) | klbs | 0–79 | Confirm abbreviation/scale in workbook. |
| Rotation | Drill-string rotation speed | RPM | 0–457 | Article notes that surface and downhole motor revolutions may both appear. |
| Torque | Drilling torque | klbs-ft | 0–12.3 | Verify exact scaling and source. |
| StandpipePressure | Standpipe pressure | psi | 9–3,915 | Review operating mode before defining any threshold rule. |
| FlowIn | Drilling fluid flow entering the well | gpm | <10–1,934 | Volumetric flow unit according to the published table. |
| FlowOut | Flow-out sensor value | % | 0–100 | **Percentage, not gpm, according to the published table. Do not calculate FlowIn − FlowOut directly.** Confirm what 0–100% represents in the original sensor/report schema. |
| PumpStroke | Pump stroke rate | spm | 6–242 | Check whether it is stroke count/rate and confirm the unit label in the workbook. |
| MudWeight | Drilling mud weight / density | pcf | 66–127 | pcf = pounds per cubic foot. Confirm source conventions. |
| FunnelViscosity | Funnel viscosity | sec/qt (article typography: “seq/qt”) | 29–76 | Confirm the intended “seconds per quart” unit from the source workbook/report. |
| PlasticViscosity | Plastic viscosity | cP | 3–48 | cP = centipoise. |
| YieldPoint | Yield point | lb/100 ft² | 2–47 | Verify the exact convention used in the reporting source. |
| Gel Strength10sec | Gel strength after 10 seconds | lb/100 ft² | 1–26 | Name spelling/spacing may vary. Match columns by normalized name only after checking. |
| Gel Strength10min | Gel strength after 10 minutes | lb/100 ft² | 1–27 | Name spelling/spacing may vary. |
| Solid | Solids content | % | 4–65 | Confirm whether this is percent by volume, mass, or another reporting convention. |

## Target column

| Column | Meaning | Published categories / thresholds | Validation note |
|---|---|---|---|
| LossesSeverity | Recorded lost-circulation severity label assigned using report-derived loss information | No loss: 0 bbl/hr; Seepage: <10 bbl/hr; Partial: 10–100 bbl/hr; Severe: >100 bbl/hr; Complete: no drilling-mud return at surface | The article states No Loss is coded 0 and Complete Loss is coded 5. Verify the actual encoding and spelling in the specific workbook. Do not infer an entire label map from two codes. |

## Missing fields needed for a stronger evaluation

The published 18-column feature/target list used here does not contain a clearly identified:
- Well ID linking each row to one physical well.
- Timestamp or verified chronological index.
- Loss onset/end time separate from the label recorded for a depth row.
- Intervention log indicating changes ordered by the drilling team.

Do not invent these fields by treating HoleSection as Well ID or mud-property signatures as report days. Depth-reset segments can be explored as a hypothesis only and must be verified against the source data.

## Derived rule warning

Because FlowIn is published in gpm and FlowOut is published as %, their direct difference is dimensionally invalid. A pressure/flow comparator requires an engineer-reviewed definition of the FlowOut percentage, any calibration to comparable flow units, sensor reliability checks, and an agreed operating context. Do not promote an arbitrary combination of pressure and percentage into an alarm rule.
