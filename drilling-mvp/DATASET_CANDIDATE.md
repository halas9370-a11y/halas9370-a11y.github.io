# Candidate dataset for HalaLoya AI Drilling Assistant

## Candidate source

Mardanirad, S., Wood, D. A., & Zakeri, H. (2021). *The application of deep learning algorithms to classify subsurface drilling lost circulation severity in large oil field datasets*. SN Applied Sciences, 3, 785. https://doi.org/10.1007/s42452-021-04769-0

Official article page: https://link.springer.com/article/10.1007/s42452-021-04769-0

The publisher lists an electronic supplementary XLSX file (about 8,024 KB) on the article page. Direct supplement URL referenced by the publisher page:
https://media.springernature.com/original/springer-static/esm/art%3A10.1007%2Fs42452-021-04769-0/MediaObjects/42452_2021_4769_MOESM1_ESM.xlsx

## Reported characteristics

The paper describes a dataset from 20 wells in the Azadegan oilfield, Iran, with 65,376 records and five fluid-loss severity classes. A related 2022 paper reports 17 predictor variables in its processed version and describes severe class imbalance. See: https://link.springer.com/article/10.1007/s13202-021-01411-y

## License and permitted use — verify before reuse

The article page states that the article is licensed under Creative Commons Attribution 4.0 International (CC BY 4.0), and it lists the XLSX as electronic supplementary material. Before redistributing the supplementary file, publishing a derived dataset, or using it commercially, verify that the supplement itself is covered by that license and check any explicit third-party notices. If uncertain, seek written clarification from the authors or publisher. Keep attribution and change notices with any permitted reuse.

## Research limits and handling notes

- This is a candidate for reproducibility and research exploration, not a ready-made validated model for other fields or companies.
- Document all column definitions, units, class definitions, and any preprocessing before modelling.
- The published studies discuss strong class imbalance, especially for the most severe loss class.
- Avoid random row splits that can place records from the same well in both training and test sets. Prefer well-level separation to test generalization to unseen wells, when identifiers are available.
- Report precision, recall, per-class metrics, confusion matrix, false alarms, and any supported lead-time measure; do not rely on overall accuracy alone.
- Do not claim operational readiness or connect outputs to real-time drilling decisions. Qualified drilling professionals must review any future research results.
- Do not upload confidential company data to the public prototype. Use only data with clear authorization and approved handling controls.

## Current project status

The current app is an exploratory data-quality tool. It does not train or run a lost-circulation prediction model. The candidate file has not yet been downloaded, parsed, or validated inside this project; its workbook schema and target-column mapping still need inspection.
