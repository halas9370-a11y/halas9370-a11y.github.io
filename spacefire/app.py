"""HalaLoya SpaceFire Explorer — exploratory data-audit prototype.

This app does not infer fire risk or train a predictive model. It helps inspect
uploaded CSV data before a scientifically justified analysis is chosen.
"""
import io
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="HalaLoya SpaceFire Explorer", page_icon="🚀", layout="wide")

st.title("HalaLoya SpaceFire Explorer")
st.caption("Open space-science data • exploratory prototype • not an operational safety tool")

st.info(
    "Research status: this prototype audits and visualizes a CSV you provide. "
    "It does not currently connect directly to NASA, validate a scientific hypothesis, "
    "or predict fire behavior. Confirm the source, license, units, and documentation before analysis."
)

with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
    st.markdown("**Candidate source to review**")
    st.link_button("NASA PSI data portal", "https://psi.nasa.gov/physci/repo/data/investigations/PSI-25")
    st.link_button("NASA Space Apps 2026", "https://www.spaceappschallenge.org/2026/")

if uploaded is None:
    st.subheader("Start with a data audit")
    st.markdown(
        "1. Download a data file from a documented source.\n"
        "2. Record the exact file name, source URL, access date, license, and units.\n"
        "3. Upload the CSV here to inspect its structure.\n"
        "4. Use the documentation to identify which columns represent experimental conditions or outcomes."
    )
    st.warning("No dataset is bundled with this app. No NASA dataset has been loaded or verified yet.")
    st.stop()

try:
    raw_bytes = uploaded.getvalue()
    df = pd.read_csv(io.BytesIO(raw_bytes), low_memory=False)
except Exception as exc:
    st.error(f"Could not read this file as a CSV: {exc}")
    st.stop()

if df.empty:
    st.warning("The uploaded file contains no data rows.")
    st.stop()

st.subheader("1. Dataset overview")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Rows", f"{len(df):,}")
m2.metric("Columns", f"{len(df.columns):,}")
m3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
m4.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")

st.caption("These are structural checks only. They do not establish that the data are scientifically valid.")
st.dataframe(df.head(25), use_container_width=True)

st.subheader("2. Column audit")
audit = pd.DataFrame({
    "column": df.columns.astype(str),
    "dtype": [str(t) for t in df.dtypes],
    "non_null": [int(df[c].notna().sum()) for c in df.columns],
    "missing": [int(df[c].isna().sum()) for c in df.columns],
    "missing_percent": [round(float(df[c].isna().mean() * 100), 2) for c in df.columns],
    "unique_values": [int(df[c].nunique(dropna=True)) for c in df.columns],
})
st.dataframe(audit, use_container_width=True, hide_index=True)

numeric_cols = df.select_dtypes(include="number").columns.tolist()
if numeric_cols:
    st.subheader("3. Explore numeric variables")
    left, right = st.columns(2)
    with left:
        chart_kind = st.selectbox("Chart type", ["Scatter", "Histogram", "Box plot", "Line"])
    with right:
        x_col = st.selectbox("X / variable", numeric_cols, index=0)
        y_options = [c for c in numeric_cols if c != x_col]
        y_col = st.selectbox("Y variable (for scatter/line)", ["— none —"] + y_options)

    if chart_kind == "Scatter":
        if y_col != "— none —":
            plot_df = df[[x_col, y_col]].dropna()
            if len(plot_df):
                fig = px.scatter(plot_df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No complete rows are available for these two columns.")
        else:
            st.info("Choose a Y variable to draw a scatter plot.")
    elif chart_kind == "Histogram":
        fig = px.histogram(df, x=x_col, title=f"Distribution of {x_col}")
        st.plotly_chart(fig, use_container_width=True)
    elif chart_kind == "Box plot":
        fig = px.box(df, y=x_col, title=f"Distribution and spread of {x_col}")
        st.plotly_chart(fig, use_container_width=True)
    elif chart_kind == "Line":
        if y_col != "— none —":
            plot_df = df[[x_col, y_col]].dropna().sort_values(x_col)
            if len(plot_df):
                fig = px.line(plot_df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No complete rows are available for these two columns.")
        else:
            st.info("Choose a Y variable to draw a line chart.")
    st.subheader("Descriptive statistics")
    st.dataframe(df[numeric_cols].describe().T, use_container_width=True)
else:
    st.info("No numeric columns were detected. Check whether numeric values are stored as text or whether this file is suitable for numeric analysis.")

st.subheader("4. Research notes")
st.text_area(
    "Record the source, units, meaning of columns, and limitations before interpreting results.",
    placeholder="Source URL, exact file/version, license, access date, units, experimental context, possible repeated measurements…",
    key="research_notes",
)
st.caption(
    "Interpretation rule: a visible relationship is not proof of causation. "
    "Do not label a column as a fire outcome without checking the source documentation."
)
