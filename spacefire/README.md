# HalaLoya SpaceFire Explorer

**Project status:** early research prototype / data-audit scaffold. Not a validated fire-risk model and not an official NASA product.

## Research question

What relationships, if any, can be observed between experimental conditions (for example, oxygen concentration, airflow, and material properties) and documented flame behavior in publicly available microgravity-combustion experiment data?

The exact variables and outcomes must be determined from source files and supporting documentation. The app does not assume that every variable exists, infer causation from correlation, or claim prediction without suitable labels and validation.

## Current prototype capabilities

- Upload a CSV/TSV or paste tabular text with a header row.
- Inspect row/column counts, missingness, duplicate rows, data types, and per-column completeness.
- Apply broad plausibility screens to oxygen and CO/CO2 readings; flags are prompts for review, not automatic corrections.
- Filter material descriptions containing PMMA.
- Calculate descriptive final-minus-initial deltas for oxygen, carbon dioxide, and carbon monoxide when the relevant columns are present.
- Explore a selected measurement and export the flagged rows or PMMA subset with derived deltas.
- Preserve the original values for comparison and review.

The app intentionally does **not** train a predictive model. Fan/air settings may contain sequences and operational notes rather than one numeric value. The experimental table does not by itself provide a verified, consistent flame-spread/extinction outcome label for every row.

## Data source

- NASA Physical Sciences Informatics (PSI), PSI-25 BASS-II: https://psi.nasa.gov/physci/repo/data/investigations/PSI-25
- NASA BASS-II Summary Report: https://ntrs.nasa.gov/citations/20210011385
- NASA BASS-II combustion results: https://ntrs.nasa.gov/citations/20160000593

The dataset was shared as text in the conversation, but it has **not** been bundled into this repository or independently parsed end-to-end. To run the app on the full table, paste the header and rows into the sidebar or upload the original CSV. Before analysis, record the exact dataset version, access date, license/terms, units, provenance, and limitations.

## First analysis plan

1. Inspect the source table and supporting scientific documentation.
2. Audit file format, columns, units, row counts, missingness, duplicates, and implausible values.
3. Identify which fields represent experimental conditions and documented outcomes.
4. Select one narrow research question that the data can answer.
5. Produce descriptive plots and summaries, showing sample sizes and limitations.
6. Check repeated measurements and avoid treating correlated rows as independent evidence.
7. Only if suitable outcome labels and enough independent observations exist, define a baseline model and an honest validation strategy.
8. Publish reproducible steps, source citations, and a plain-language summary.

## Limitations

- No predictive model has been trained or validated.
- Plausibility flags are broad screening checks, not scientific corrections.
- Derived deltas are sensor-reading differences, not flame-spread rates or direct measures of fire danger.
- This is not an operational fire-safety tool.
- The project is independent and is not affiliated with or endorsed by NASA.
- Verify official 2026 event wording, eligibility, local logistics, and submission requirements at https://www.spaceappschallenge.org/2026/.

## Repository structure

- `app.py`: exploratory data audit, PMMA subset, deltas, plots, and exports.
- `requirements.txt`: runtime dependencies.
- `README.md`: project framing, sources, methods, and limitations.

## Change log

- Initial scaffold created 2026-10-20.
- Updated 2026-10-21: added pasted-table support, PMMA exploratory analysis, quality flags, and export options.
