# Deploy SpaceFire Explorer on Streamlit Community Cloud

The HalaLoya GitHub Pages site is static HTML and cannot run this Python app directly. Deploy this folder as a separate Streamlit app.

1. Sign in at https://share.streamlit.io/ using the GitHub account that can access this repository.
2. Choose **Create app** / **Deploy an app**.
3. Select repository: `halas9370-a11y/halas9370-a11y.github.io`.
4. Branch: `main`.
5. Main file path: `spacefire/app.py`.
6. If prompted for dependencies, use `spacefire/requirements.txt`. Streamlit Cloud may detect this file automatically because it is alongside the app.
7. Deploy, then test with a small, non-sensitive CSV first.
8. Copy the resulting Streamlit app URL into the HalaLoya research portfolio only after deployment succeeds.

## Before publishing

- Do not upload confidential, personal, or restricted datasets.
- Confirm the exact data source and reuse terms before sharing results.
- Test the app with an ordinary CSV containing numeric and text columns.
- The app currently provides descriptive inspection and exploratory plots only. It is not a fire-risk predictor and is not validated for safety decisions.
- If deployment reports a dependency or runtime error, record the full error message and fix it before linking the live app publicly.
