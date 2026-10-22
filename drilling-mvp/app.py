import io
import json
from datetime import datetime, timezone

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="HalaLoya AI Drilling Assistant", page_icon="🛢️", layout="wide")

st.title("HalaLoya AI Drilling Assistant")
st.caption("Early research prototype · CSV/Excel exploration and data-quality checks")

st.warning(
    "Research use only. This prototype does not predict lost circulation, "
    "is not validated for drilling operations, and must not be used for safety-critical decisions. "
    "This is a public hosted app: do not upload confidential or commercially sensitive company data."
)

with st.expander("Before uploading data", expanded=True):
    st.markdown(
        "- Upload only data you are authorized to use.\n"
        "- Do not upload confidential, commercially sensitive, personal, or regulated information to this public prototype.\n"
        "- Verify the dataset license and attribution requirements before research publication.\n"
        "- Results are exploratory statistics, not engineering conclusions."
    )

uploaded = st.file_uploader("Upload a drilling dataset (CSV or Excel XLSX)", type=["csv", "xlsx"])

if uploaded is None:
    st.info("Upload a CSV or XLSX file to inspect its structure and explore numeric variables.")
    st.stop()

raw = uploaded.getvalue()
source_sheet = None
try:
    if uploaded.name.lower().endswith(".csv"):
        df = pd.read_csv(io.BytesIO(raw), low_memory=False)
    elif uploaded.name.lower().endswith(".xlsx"):
        workbook = pd.ExcelFile(io.BytesIO(raw), engine="openpyxl")
        if not workbook.sheet_names:
            st.error("No worksheets were found in this Excel workbook.")
            st.stop()
        source_sheet = st.selectbox("Choose an Excel worksheet", workbook.sheet_names)
        df = pd.read_excel(workbook, sheet_name=source_sheet)
    else:
        st.error("Unsupported file type. Upload a CSV or XLSX file.")
        st.stop()
except Exception as exc:
    st.error(f"Could not read this file. Check the file format, encoding, and worksheet structure. Details: {exc}")
    st.stop()

if df.empty:
    st.error("The CSV contains no data rows.")
    st.stop()

st.subheader("Dataset overview")
a, b, c, d = st.columns(4)
a.metric("Rows", f"{len(df):,}")
b.metric("Columns", f"{len(df.columns):,}")
c.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")
missing_total = int(df.isna().sum().sum())
d.metric("Missing cells", f"{missing_total:,}")

st.subheader("Column summary")
summary = pd.DataFrame({
    "column": df.columns.astype(str),
    "data_type": [str(df[col].dtype) for col in df.columns],
    "missing_count": [int(df[col].isna().sum()) for col in df.columns],
    "missing_percent": [round(float(df[col].isna().mean() * 100), 2) for col in df.columns],
    "unique_values": [int(df[col].nunique(dropna=True)) for col in df.columns],
})
st.dataframe(summary, use_container_width=True, hide_index=True)

st.subheader("Preview")
st.dataframe(df.head(100), use_container_width=True)

numeric_cols = df.select_dtypes(include="number").columns.tolist()
if numeric_cols:
    st.subheader("Exploratory chart")
    x_options = ["Row order"] + [str(col) for col in df.columns]
    x_choice = st.selectbox("X-axis", x_options, index=0)
    y_choice = st.selectbox("Numeric variable (Y-axis)", numeric_cols)
    chart_df = df.copy()
    if x_choice == "Row order":
        chart_df["_row_order_"] = range(1, len(chart_df) + 1)
        x_col = "_row_order_"
        x_label = "Row order (not necessarily time or measured depth)"
    else:
        x_col = x_choice
        x_label = x_choice

    chart_df = chart_df[[x_col, y_choice]].dropna()
    if len(chart_df) > 5000:
        chart_df = chart_df.sample(5000, random_state=42).sort_index()
        st.caption("Showing a reproducible sample of up to 5,000 non-missing rows for responsiveness.")
    fig = px.scatter(chart_df, x=x_col, y=y_choice, title=f"{y_choice} by {x_label}")
    fig.update_layout(xaxis_title=x_label, yaxis_title=str(y_choice))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No numeric columns were detected, so a numeric chart cannot be created.")

st.subheader("Optional statistical screening")
st.caption("IQR flags are simple statistical outliers only. They are not lost-circulation events or risk predictions.")
if numeric_cols:
    selected = st.selectbox("Select a numeric column", numeric_cols, key="iqr_col")
    values = pd.to_numeric(df[selected], errors="coerce").dropna()
    if len(values) >= 4:
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        flagged = ((values < lower) | (values > upper))
        st.write(f"IQR bounds: **{lower:.6g}** to **{upper:.6g}**")
        st.write(f"Values flagged by this rule: **{int(flagged.sum()):,}** of {len(values):,}")
    else:
        st.info("At least four non-missing numeric values are needed for this screening.")

report = {
    "project": "HalaLoya AI Drilling Assistant",
    "report_generated_utc": datetime.now(timezone.utc).isoformat(),
    "uploaded_filename": uploaded.name,
    "source_worksheet": source_sheet,
    "row_count": int(len(df)),
    "column_count": int(len(df.columns)),
    "duplicate_row_count": int(df.duplicated().sum()),
    "missing_cell_count": missing_total,
    "columns": summary.to_dict(orient="records"),
    "limitations": [
        "Exploratory data-quality report only.",
        "No lost-circulation prediction or model training is performed.",
        "Not validated for operational or safety-critical use.",
        "Dataset license and permission must be verified independently."
    ]
}
st.download_button(
    "Download data-quality report (JSON)",
    data=json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8"),
    file_name="halaloya_data_quality_report.json",
    mime="application/json",
)
