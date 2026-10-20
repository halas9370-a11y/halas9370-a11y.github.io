# HalaLoya SpaceFire Explorer

**Project status:** early research prototype / data-audit scaffold. Not a validated fire-risk model and not an official NASA product.

## Research question

What relationships, if any, can be observed between experimental conditions (for example, oxygen concentration, airflow, and material properties) and documented flame behavior in publicly available microgravity-combustion experiment data?

The exact variables and outcomes will be determined by inspecting the source files. We will not assume that every variable exists, infer causation from correlations, or claim prediction unless the data and validation support it.

## Why this project

The project aims to make open space-science data easier to inspect and discuss through:
- a reproducible data-quality audit;
- transparent charts and descriptive statistics;
- source-linked notes and limitations;
- a bilingual, accessible interface as the project matures.

## Data source to assess

- NASA Physical Sciences Informatics (PSI): https://psi.nasa.gov/physci/repo/data/investigations/PSI-25
- NASA Technical Reports Server search/record: https://ntrs.nasa.gov/citations/20160000593

These are candidate starting points. Before analysis or redistribution, record the exact dataset/file title, version, access date, license/terms, units, provenance, and any restrictions. Do not assume that all material linked from a NASA page has identical reuse terms.

## First analysis plan

1. Inventory available files and document their source and terms.
2. Inspect file formats, columns, units, row counts, missingness, duplicates, and obvious invalid values.
3. Identify which fields actually represent experimental conditions and outcomes; consult the accompanying scientific documentation.
4. Choose one narrow question that the available observations can answer.
5. Produce exploratory plots and descriptive summaries with uncertainty and sample sizes shown where possible.
6. Check whether observations are repeated measurements from the same experiment; avoid treating correlated rows as independent evidence.
7. Only if there are suitable outcome labels and enough independent observations, define a baseline model and an honest validation strategy. Compare against a simple baseline and report limitations.
8. Publish reproducible steps, source citations, and a plain-language summary.

## MVP acceptance criteria

- A user can upload a CSV file and see its dimensions, columns, data types, missing values, and duplicate-row count.
- A user can choose numeric columns and generate a basic plot.
- The app does not silently alter the source data.
- Every conclusion is tied to inspected data and documented limitations.
- No claim of fire prediction, operational safety, or NASA endorsement is made.

## Current limitations

- The actual downloadable files and their schema have not yet been verified in this repository.
- No model has been trained or validated.
- This is not an operational fire-safety tool.
- The project is independent and is not affiliated with or endorsed by NASA.
- Before entering a NASA Space Apps event, verify the exact official 2026 challenge wording, eligibility, local event logistics, and submission requirements at https://www.spaceappschallenge.org/2026/.

## Suggested team roles

- Research lead (Hala): literature review, question framing, source tracking, limitations, bilingual explanation, and user experience.
- Data analyst: dataset inspection, reproducible analysis, charts, and statistical review.
- Developer: app usability, testing, and deployment.

## Repository structure

- `app.py`: exploratory CSV audit and plotting prototype.
- `requirements.txt`: minimal runtime dependencies.
- `README.md`: research question, source assessment, and methodology.

## Change log

- Initial scaffold created 2026-10-20.
