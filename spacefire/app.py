"""HalaLoya SpaceFire Explorer — NASA BASS-II data-audit prototype.

Exploratory analysis only. It does not infer fire risk or train a predictive model.
"""
import io
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="HalaLoya SpaceFire Explorer", page_icon="🚀", layout="wide")
st.title("HalaLoya SpaceFire Explorer")
st.caption("NASA PSI-25 / BASS-II • research prototype • not an operational fire-safety tool")

st.info(
    "This tool audits source data and explores measured gas changes. It does not predict "
    "flame spread/extinction. Missing or suspicious values are flagged, never silently corrected."
)

NASA_URL = "https://psi.nasa.gov/physci/repo/data/investigations/PSI-25"
NUMERIC_COLUMNS = [
    "GMT", "Sample #", "Flow restrictor", "Total Frames Shot",
    "Calibrated initial O2 % by vol", "Calibrated final O2 % by vol",
    "Initial CO2 % by vol", "Final CO2 % by vol",
    "Initial CO (ppm)", "Final CO (ppm)",
]

with st.sidebar:
    st.header("Source")
    st.link_button("Open NASA PSI-25 record", NASA_URL)
    st.markdown("**Input options**")
    uploaded = st.file_uploader("Upload original CSV", type=["csv", "tsv", "txt"])
    pasted = st.text_area(
        "Or paste the table text here",
        height=220,
        placeholder="Paste the header row and data rows, separated by tabs or commas…",
    )
    st.caption("Keep the original headers and values. Do not manually fix questionable readings.")

df = None
source_label = ""
if uploaded is not None:
    try:
        raw = uploaded.getvalue()
        try:
            df = pd.read_csv(io.BytesIO(raw), sep=None, engine="python", low_memory=False)
        except Exception:
            df = pd.read_csv(io.BytesIO(raw), low_memory=False)
        source_label = uploaded.name
    except Exception as exc:
        st.error(f"Could not parse uploaded file: {exc}")
elif pasted.strip():
    try:
        df = pd.read_csv(io.StringIO(pasted.strip()), sep=None, engine="python", low_memory=False)
        source_label = "Pasted table text"
    except Exception as exc:
        st.error(f"Could not parse pasted text. Please paste the header and rows separated by tabs or commas. Details: {exc}")

if df is None:
    st.subheader("Ready for the BASS-II table")
    st.markdown(
        "1. Open the NASA PSI-25 record.\n"
        "2. Paste the tabular text in the sidebar, or upload the original CSV when available.\n"
        "3. Review the audit flags before interpreting any relationships."
    )
    st.warning("No dataset has been loaded into this app yet. The table shared in chat has not been written into the repository.")
    st.stop()

if df.empty:
    st.warning("No data rows were detected.")
    st.stop()

df.columns = [str(c).strip() for c in df.columns]
raw_df = df.copy()
st.caption(f"Input: {source_label}. Raw values are retained in the displayed source table.")

# Convert only known measurement columns. Preserve the original values separately.
numeric_df = df.copy()
for col in NUMERIC_COLUMNS:
    if col in numeric_df.columns:
        numeric_df[col] = pd.to_numeric(
            numeric_df[col].astype(str).str.strip().replace(
                {"": pd.NA, "NaN": pd.NA, "nan": pd.NA, "none": pd.NA, "None": pd.NA}
            ),
            errors="coerce",
        )

st.subheader("1. Dataset overview")
a, b, c, d = st.columns(4)
a.metric("Rows", f"{len(df):,}")
b.metric("Columns", f"{len(df.columns):,}")
c.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
d.metric("Exact duplicate rows", f"{int(df.duplicated().sum()):,}")
st.dataframe(raw_df.head(25), use_container_width=True, hide_index=True)

st.subheader("2. Data-quality audit")
audit = pd.DataFrame({
    "column": df.columns,
    "detected_type": [str(t) for t in numeric_df.dtypes],
    "non_missing": [int(numeric_df[col].notna().sum()) for col in df.columns],
    "missing": [int(numeric_df[col].isna().sum()) for col in df.columns],
    "missing_percent": [round(float(numeric_df[col].isna().mean() * 100), 2) for col in df.columns],
    "unique_values": [int(df[col].nunique(dropna=True)) for col in df.columns],
})
st.dataframe(audit, use_container_width=True, hide_index=True)

flags = []
for i, row in numeric_df.iterrows():
    test = str(df.iloc[i].get("Test #", f"row {i + 1}"))
    for col in ["Calibrated initial O2 % by vol", "Calibrated final O2 % by vol"]:
        if col in numeric_df.columns and pd.notna(row.get(col)):
            value = row[col]
            if value < 0 or value > 25:
                flags.append({"Test #": test, "Column": col, "Value": value,
                              "Reason": "Outside broad 0–25% plausibility screen; verify source record"})
    for col in ["Initial CO2 % by vol", "Final CO2 % by vol"]:
        if col in numeric_df.columns and pd.notna(row.get(col)):
            value = row[col]
            if value < 0 or value > 10:
                flags.append({"Test #": test, "Column": col, "Value": value,
                              "Reason": "Outside broad 0–10% screening range; verify units/source"})
    for col in ["Initial CO (ppm)", "Final CO (ppm)"]:
        if col in numeric_df.columns and pd.notna(row.get(col)) and row[col] < 0:
            flags.append({"Test #": test, "Column": col, "Value": row[col],
                          "Reason": "Negative sensor reading; preserve but verify instrument convention"})

if flags:
    st.warning(f"{len(flags)} potential data-quality flag(s). These are review prompts, not automatic corrections.")
    st.dataframe(pd.DataFrame(flags), use_container_width=True, hide_index=True)
    st.download_button(
        "Download quality flags CSV",
        pd.DataFrame(flags).to_csv(index=False).encode("utf-8-sig"),
        file_name="spacefire_quality_flags.csv",
        mime="text/csv",
    )
else:
    st.success("No flags triggered by the current broad screening rules.")

st.subheader("3. Exploratory PMMA subset")
if "Fuel Sample Material" in df.columns:
    pmma_mask = df["Fuel Sample Material"].astype(str).str.contains("PMMA", case=False, na=False)
    pmma_raw = df.loc[pmma_mask].copy()
    pmma = numeric_df.loc[pmma_mask].copy()
    st.write(f"Rows whose material description contains PMMA: **{len(pmma):,}**")
    if len(pmma):
        delta_map = {
            "Delta O2 (percentage points)": ("Calibrated final O2 % by vol", "Calibrated initial O2 % by vol"),
            "Delta CO2 (percentage points)": ("Final CO2 % by vol", "Initial CO2 % by vol"),
            "Delta CO (ppm)": ("Final CO (ppm)", "Initial CO (ppm)"),
        }
        for new_col, (final_col, initial_col) in delta_map.items():
            if final_col in pmma.columns and initial_col in pmma.columns:
                pmma[new_col] = pmma[final_col] - pmma[initial_col]

        show_cols = [c for c in ["PI", "Test #", "Fuel Sample Material", "Calibrated initial O2 % by vol",
                                  "Calibrated final O2 % by vol", "Delta O2 (percentage points)",
                                  "Initial CO2 % by vol", "Final CO2 % by vol",
                                  "Delta CO2 (percentage points)", "Initial CO (ppm)", "Final CO (ppm)",
                                  "Delta CO (ppm)"] if c in pmma.columns]
        st.dataframe(pmma[show_cols], use_container_width=True, hide_index=True)
        st.download_button(
            "Download PMMA subset with derived deltas",
            pmma.to_csv(index=False).encode("utf-8-sig"),
            file_name="spacefire_pmma_exploratory.csv",
            mime="text/csv",
        )

        choices = [c for c in ["Delta O2 (percentage points)", "Delta CO2 (percentage points)", "Delta CO (ppm)",
                                "Calibrated initial O2 % by vol", "Final CO (ppm)"] if c in pmma.columns]
        if choices:
            chosen = st.selectbox("Plot a PMMA measurement", choices)
            plot_data = pmma[["Test #" if "Test #" in pmma.columns else pmma.index.name or chosen, chosen]].copy()
            if "Test #" not in pmma.columns:
                plot_data = pd.DataFrame({"row": pmma.index.astype(str), chosen: pmma[chosen].values})
                x_key = "row"
            else:
                x_key = "Test #"
            plot_data = plot_data.dropna(subset=[chosen])
            if len(plot_data):
                fig = px.scatter(plot_data, x=x_key, y=chosen, title=f"PMMA exploratory view: {chosen}")
                st.plotly_chart(fig, use_container_width=True)
        st.caption("PMMA is a descriptive material filter only. Tests may differ in geometry, flow, oxygen, date, and procedure; do not compare them as controlled replicates without checking the documentation.")
    else:
        st.info("No material descriptions containing PMMA were detected.")
else:
    st.info("Could not find the expected 'Fuel Sample Material' column. Check that the original header row was included.")

st.subheader("4. Interpretation limits")
st.markdown(
    "- A delta is final minus initial sensor reading; it is not a flame-spread rate.\n"
    "- The fan/air columns can contain sequences and notes, so they must not be treated as single numeric settings.\n"
    "- A flagged value is not necessarily wrong; consult the NASA record and supporting documentation.\n"
    "- No predictive model is trained here because this table alone does not provide a verified, consistent outcome label for every test."
)
st.link_button("NASA BASS-II Summary Report", "https://ntrs.nasa.gov/citations/20210011385")
st.caption("Research prototype only. Not for operational fire-safety decisions.")
