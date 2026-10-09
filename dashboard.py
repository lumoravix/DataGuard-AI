"""SaaS-style Streamlit presentation layer for DataGuard AI."""

from __future__ import annotations

import hashlib

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.anomaly_detector import DEFAULT_CONTAMINATION, detect_anomalies
from src.cleaner import clean_dataframe, dataframe_to_csv_bytes
from src.csv_loader import CSVLoadError, DEFAULT_MAX_FILE_SIZE_MB, load_csv
from src.profiler import missing_values_table, profile_dataframe
from src.quality_score import calculate_quality_score
from src.report_generator import generate_html_report
from src.validator import validate_dataframe


NAVIGATION = [
    "Overview & Upload", "Data Profiling", "Data Validation", "Anomaly Detection",
    "Quality Score", "Data Cleaning & Reports",
]


def _apply_theme() -> None:
    st.markdown("""<style>
    .stApp {background: radial-gradient(circle at 15% 0%, #172554 0%, #090f22 42%, #060914 100%); color:#e5e7eb;}
    [data-testid="stMainBlockContainer"], .main .block-container {max-width:1480px; padding:5.5rem 2.2rem 3.5rem;}
    [data-testid="stSidebar"] {background:linear-gradient(180deg,#0e1630,#090d1d); border-right:1px solid #253456;}
    [data-testid="stSidebar"] * {color:#e6edff;}
    [data-testid="stSidebar"] h2 {font-size:1.85rem!important; line-height:1.15; font-weight:800; letter-spacing:-.025em; margin:0 0 .35rem!important;}
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {font-size:.92rem!important; line-height:1.45; letter-spacing:.045em; color:#b9adff!important;}
    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] {gap:.4rem;}
    [data-testid="stSidebar"] [data-testid="stRadio"] label {font-size:1.15rem!important; font-weight:600; line-height:1.35; padding:.3rem 0; align-items:flex-start;}
    [data-testid="stSidebar"] [data-testid="stRadio"] label > div {margin-top:.14rem; flex:0 0 auto;}
    [data-testid="stSidebar"] [data-testid="stFileUploader"] label {font-size:1rem!important; font-weight:650; line-height:1.4;}
    [data-testid="stSidebar"] [data-testid="stFileUploader"] small,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] {font-size:.95rem!important; line-height:1.45;}
    h1,h2,h3 {color:#f5f3ff!important; letter-spacing:-.02em; margin-top:1.35rem!important;}
    h2 {font-size:1.35rem!important;} h3 {font-size:1.08rem!important;}
    [data-testid="stMetric"] {background:linear-gradient(145deg,rgba(27,40,75,.94),rgba(15,24,48,.94)); border:1px solid #324775; border-radius:14px; padding:16px 18px; min-height:112px; display:flex; flex-direction:column; justify-content:center; box-shadow:0 12px 24px rgba(0,0,0,.16);}
    [data-testid="stMetricLabel"] {color:#aab9dc!important;} [data-testid="stMetricValue"] {color:#f4f1ff!important;}
    .dg-eyebrow {color:#a99bff; font-size:.72rem; font-weight:750; letter-spacing:.14em; text-transform:uppercase;}
    .dg-title {font-size:clamp(2.125rem,3.25vw,2.875rem); line-height:1.12; font-weight:780; letter-spacing:-.035em; margin:.3rem 0 .65rem; color:#f8fafc; overflow-wrap:anywhere;}
    .dg-subtitle {color:#b4c1e3; font-size:clamp(.92rem,1.4vw,1.04rem); max-width:760px; line-height:1.55; margin-bottom:1.75rem;}
    .dg-panel {background:rgba(15,24,48,.7); border:1px solid #293a65; border-radius:14px; padding:1rem 1.2rem; margin:.5rem 0 1rem;}
    .stButton>button, .stDownloadButton>button {border-radius:9px; border:1px solid #7061d6; background:linear-gradient(135deg,#6554cc,#4d40ac); color:white; font-weight:650; min-height:2.55rem; padding:0 .95rem;}
    .stButton>button:hover, .stDownloadButton>button:hover {background:#7061d6; border-color:#9f8cff;}
    [data-testid="stDataFrame"] {border:1px solid #293a65; border-radius:10px; overflow:hidden;}
    [data-testid="stExpander"] {border:1px solid #293a65; border-radius:12px; background:rgba(15,24,48,.55);}
    [data-testid="stAlert"] {border-radius:10px; border-width:1px;}
    [data-testid="stPlotlyChart"] {border:1px solid #25375f; border-radius:14px; padding:4px; background:rgba(8,14,30,.35);}
    div[data-baseweb="select"] > div, .stTextInput input {background:#101a35!important; border-color:#34466f!important;}
    @media (max-width: 900px) {
      [data-testid="stMainBlockContainer"], .main .block-container {padding:4.75rem 1rem 2rem;}
      [data-testid="stMetric"] {min-height:96px; padding:13px;}
      .dg-title {font-size:clamp(1.9rem,5vw,2.25rem); margin-top:.2rem;}
    }
    @media (max-width: 640px) {
      [data-testid="stMainBlockContainer"], .main .block-container {padding:4.3rem .8rem 1.5rem;}
      .dg-title {font-size:clamp(1.7rem,8vw,2rem); line-height:1.16; margin-bottom:.5rem;}
      .dg-subtitle {margin-bottom:1.15rem;}
    }
    </style>""", unsafe_allow_html=True)


def _chart_layout(figure, title: str, height: int = 330):
    figure.update_layout(
        title=title, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(12,20,43,.5)", font={"color": "#dbe7ff"},
        height=height, autosize=True, hoverlabel={"bgcolor": "#121d3d", "font": {"color": "#f8fafc"}},
        margin={"l": 28, "r": 24, "t": 58, "b": 42},
        title_font={"size": 16, "color": "#f1f5ff"},
    )
    figure.update_xaxes(gridcolor="rgba(116,139,190,.16)", zerolinecolor="rgba(116,139,190,.22)")
    figure.update_yaxes(gridcolor="rgba(116,139,190,.16)", zerolinecolor="rgba(116,139,190,.22)")
    return figure


def _heading(eyebrow: str, title: str, subtitle: str) -> None:
    st.markdown(f'<div class="dg-eyebrow">{eyebrow}</div><div class="dg-title">{title}</div><div class="dg-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def _rules() -> tuple[dict, dict]:
    return st.session_state.setdefault("numeric_ranges", {}), st.session_state.setdefault("expected_types", {})


def _current_analysis(dataframe):
    ranges, expected = _rules()
    validation = validate_dataframe(dataframe, ranges, expected)
    quality = calculate_quality_score(dataframe, ranges, expected, validation)
    return validation, quality


def _overview(dataframe, profile, quality) -> None:
    _heading("Workspace", "Overview & Upload", "Securely upload a CSV and monitor its quality in one place.")
    st.markdown('<div class="dg-panel">Your uploaded CSV remains in this browser session. DataGuard AI never modifies the source dataset.</div>', unsafe_allow_html=True)
    anomaly = st.session_state.get("anomaly_results")
    anomaly_count = anomaly["total_anomalies"] if anomaly and anomaly.get("status") == "success" else "—"
    score = f'{quality["overall_score"]}/100' if quality["status"] == "success" else "—"
    cards = st.columns(5)
    cards[0].metric("Dataset size", f'{profile["rows"]} × {profile["columns"]}')
    cards[1].metric("Missing values", sum(profile["missing_values"].values()))
    cards[2].metric("Duplicate rows", profile["duplicate_rows"])
    cards[3].metric("Anomalies", anomaly_count)
    cards[4].metric("Quality score", score)
    st.subheader("Quick preview")
    st.dataframe(dataframe.head(10), use_container_width=True, hide_index=True)


def _profiling(profile) -> None:
    _heading("Explore", "Data Profiling", "Understand dataset structure, completeness, and numerical distributions.")
    missing = missing_values_table(profile)
    top = st.columns(4)
    top[0].metric("Rows", profile["rows"]); top[1].metric("Columns", profile["columns"])
    top[2].metric("Missing", sum(profile["missing_values"].values())); top[3].metric("Duplicates", profile["duplicate_rows"])
    st.subheader("Missing values by column")
    if missing["Missing values"].sum():
        figure = px.bar(missing, x="Column", y="Missing values", color="Missing (%)", color_continuous_scale="Purples")
        st.plotly_chart(_chart_layout(figure, "Completeness by column", 340), use_container_width=True, config={"displayModeBar": False, "responsive": True})
    else:
        st.success("No missing values found in this dataset.")
    st.dataframe(missing, use_container_width=True, hide_index=True)
    st.subheader("Numerical summary")
    if profile["numerical_summary"].empty:
        st.info("No numerical columns are available for a statistical summary.")
    else:
        st.dataframe(profile["numerical_summary"], use_container_width=True)


def _validation(dataframe) -> tuple[list[dict], dict]:
    _heading("Protect", "Data Validation", "Configure optional safeguards and inspect quality issues before making changes.")
    ranges, expected = _rules()
    with st.expander("Validation rules", expanded=True):
        numeric_columns = dataframe.select_dtypes(include="number").columns.tolist()
        left, right = st.columns(2)
        with left:
            if numeric_columns:
                column = st.selectbox("Numerical column", numeric_columns, key="range_column")
                use_minimum = st.checkbox("Apply minimum", key="range_min_enabled")
                minimum = st.number_input("Minimum value", value=0.0, key="range_min") if use_minimum else None
                use_maximum = st.checkbox("Apply maximum", key="range_max_enabled")
                maximum = st.number_input("Maximum value", value=0.0, key="range_max") if use_maximum else None
                if st.button("Save numerical rule"):
                    if use_minimum or use_maximum:
                        ranges[column] = {"min": minimum, "max": maximum}
                    else:
                        ranges.pop(column, None)
            else:
                st.info("No numerical columns are available for range validation.")
        with right:
            if len(dataframe.columns):
                column = st.selectbox("Column type rule", dataframe.columns.tolist(), key="type_column")
                expected_type = st.selectbox("Expected type", ["No type rule", "number", "text", "datetime", "boolean"], key="expected_type")
                if st.button("Save type rule"):
                    if expected_type == "No type rule": expected.pop(column, None)
                    else: expected[column] = expected_type
            else:
                st.info("No columns are available for type validation.")
        if ranges or expected:
            st.caption(f"Active rules: {len(ranges)} numerical range rule(s), {len(expected)} type rule(s).")
    validation, quality = _current_analysis(dataframe)
    if validation:
        st.warning(f"{len(validation)} issue type(s) detected. Review affected row indices before deciding on corrective action.")
        display = []
        for issue in validation:
            item = issue.copy(); item["affected_row_indices"] = ", ".join(map(str, item["affected_row_indices"])); display.append(item)
        st.dataframe(display, use_container_width=True, hide_index=True)
    else:
        st.success("No issues were detected with the current validation rules.")
    return validation, quality


def _anomalies(dataframe) -> None:
    _heading("Detect", "Anomaly Detection", "Isolation Forest finds unusual patterns; unusual records are not confirmed errors.")
    enabled = st.toggle("Enable anomaly detection", key="anomaly_enabled")
    if not enabled:
        st.session_state["anomaly_results"] = None
        st.info("Enable analysis to run a reproducible Isolation Forest model on suitable numerical features.")
        return
    contamination = st.slider("Expected anomaly proportion", 0.01, 0.20, DEFAULT_CONTAMINATION, 0.01, key="contamination")
    result = detect_anomalies(dataframe, contamination)
    st.session_state["anomaly_results"] = result
    if result["status"] != "success":
        st.info(result["message"]); return
    cards = st.columns(3)
    cards[0].metric("Flagged records", result["total_anomalies"])
    cards[1].metric("Anomalous rows", f'{result["anomaly_percentage"]}%')
    cards[2].metric("Features used", len(result["numerical_features_used"]))
    st.caption(result["message"] + " Features: " + ", ".join(result["numerical_features_used"]))
    figure = px.scatter(result["row_results"], x="row_index", y="anomaly_score", color="prediction", color_discrete_map={"normal":"#60a5fa", "anomaly":"#f472b6"})
    st.plotly_chart(_chart_layout(figure, "Anomaly score by record — lower is more unusual", 350), use_container_width=True, config={"displayModeBar": False, "responsive": True})
    flagged = result["row_results"]["prediction"].eq("anomaly").to_numpy()
    if flagged.any():
        records = dataframe.iloc[flagged].copy(); records.insert(0, "Row index", dataframe.index[flagged])
        records["Anomaly score"] = result["row_results"].loc[flagged, "anomaly_score"].to_numpy()
        st.subheader("Flagged records")
        st.dataframe(records, use_container_width=True, hide_index=True)


def _quality(dataframe) -> tuple[list[dict], dict]:
    _heading("Measure", "Data Quality Score", "A transparent weighted score based on completeness, uniqueness, and configured validity rules.")
    validation, result = _current_analysis(dataframe)
    if result["status"] != "success": st.info(result["explanations"][0]); return validation, result
    component = result["components"]
    cards = st.columns(4)
    cards[0].metric("Overall quality", f'{result["overall_score"]}/100')
    cards[1].metric("Completeness", f'{component["completeness"]}%')
    cards[2].metric("Uniqueness", f'{component["uniqueness"]}%')
    cards[3].metric("Validity", "Not evaluated" if component["validity"] is None else f'{component["validity"]}%')
    gauge = go.Figure(go.Indicator(mode="gauge+number", value=result["overall_score"], number={"suffix":" / 100", "font":{"size":34}}, gauge={"axis":{"range":[0,100], "tickwidth":1, "tickcolor":"#b9c5e5"},"bar":{"color":"#8b7cf6", "thickness":0.55},"steps":[{"range":[0,50],"color":"#43233f"},{"range":[50,75],"color":"#4a3d2f"},{"range":[75,100],"color":"#203e4b"}]}))
    items = {name.title(): value for name, value in component.items() if value is not None}
    chart = px.bar(x=list(items), y=list(items.values()), color=list(items.values()), color_continuous_scale="Purples", range_y=[0,100], labels={"x":"Component","y":"Score (%)"})
    gauge_column, chart_column = st.columns([1, 1.35], gap="large")
    with gauge_column:
        st.plotly_chart(_chart_layout(gauge, "Overall data quality", 275), use_container_width=True, config={"displayModeBar": False, "responsive": True})
    with chart_column:
        st.plotly_chart(_chart_layout(chart, "Quality component comparison", 275), use_container_width=True, config={"displayModeBar": False, "responsive": True})
    st.markdown('<div class="dg-panel"><strong>Effective weights:</strong> ' + " · ".join(f"{name.title()} {weight}%" for name, weight in result["effective_weights"].items()) + "</div>", unsafe_allow_html=True)
    for line in result["explanations"]: st.write(f"• {line}")
    if result["validity_issue_explanations"]: st.warning(" ".join(result["validity_issue_explanations"]))
    return validation, result


def _clean_and_report(dataframe, profile) -> None:
    _heading("Prepare", "Data Cleaning & Reports", "Generate an opt-in cleaning preview or export a compact, safely escaped report.")
    left, right = st.columns(2)
    with left:
        numerical = st.selectbox("Numerical missing values", ["unchanged", "mean", "median"], key="clean_numeric")
    with right:
        categorical = st.selectbox("Categorical missing values", ["unchanged", "mode"], key="clean_categorical")
    duplicates = st.checkbox("Remove duplicate rows (keep first)", key="clean_duplicates")
    signature = (st.session_state["upload_hash"], numerical, categorical, duplicates)
    if st.button("Generate cleaned preview", type="primary"):
        st.session_state["cleaning_result"] = clean_dataframe(dataframe, numerical, categorical, duplicates)
        st.session_state["cleaning_signature"] = signature
    result = st.session_state.get("cleaning_result") if st.session_state.get("cleaning_signature") == signature else None
    st.subheader("Original preview")
    st.dataframe(dataframe.head(10), use_container_width=True, hide_index=True)
    if result:
        st.subheader("Cleaned preview")
        st.dataframe(result["cleaned_dataframe"].head(10), use_container_width=True, hide_index=True)
        cards = st.columns(3)
        cards[0].metric("Missing values", f'{result["before_missing"]} → {result["after_missing"]}')
        cards[1].metric("Duplicate rows", f'{result["before_duplicates"]} → {result["after_duplicates"]}')
        cards[2].metric("Rows removed", result["rows_removed"])
        for line in result["operations"] + result["notes"] + result["data_type_changes"]: st.write(f"• {line}")
        st.download_button("Download Cleaned CSV", dataframe_to_csv_bytes(result["cleaned_dataframe"]), "dataguard_cleaned.csv", "text/csv")
    validation, quality = _current_analysis(dataframe)
    report = generate_html_report(profile, quality, validation, st.session_state.get("anomaly_results"), result)
    st.divider(); st.subheader("Data Quality Report")
    st.caption("The HTML report summarizes quality metrics and never embeds the uploaded dataset rows.")
    st.download_button("Download Data Quality Report", report.encode("utf-8"), "dataguard_data_quality_report.html", "text/html")


def run_dashboard() -> None:
    st.set_page_config(page_title="DataGuard AI", page_icon="◈", layout="wide", initial_sidebar_state="expanded")
    _apply_theme()
    with st.sidebar:
        st.markdown("## ◈ DataGuard AI")
        st.caption("DATA QUALITY INTELLIGENCE")
        page = st.radio("Workspace", NAVIGATION, label_visibility="collapsed")
        st.divider()
        st.caption("CSV-only · Maximum upload size: 20 MB")
        upload = st.file_uploader("Upload dataset", type=["csv"], key="dataset_upload")
    if upload:
        content = upload.getvalue(); digest = hashlib.sha256(content).hexdigest()
        if st.session_state.get("upload_hash") != digest:
            try:
                st.session_state["dataframe"] = load_csv(upload)
                st.session_state["upload_hash"] = digest
                st.session_state["cleaning_result"] = None
                st.session_state["anomaly_results"] = None
            except CSVLoadError as error:
                st.session_state["dataframe"] = None
                st.error(str(error))
    dataframe = st.session_state.get("dataframe")
    if dataframe is None:
        _heading("Welcome", "DataGuard AI", "Upload a CSV from the sidebar to begin a secure data-quality review.")
        st.info(f"CSV files only, up to {DEFAULT_MAX_FILE_SIZE_MB} MB. Your file is handled as data and is never executed.")
        return
    profile = profile_dataframe(dataframe)
    _, quality = _current_analysis(dataframe)
    if page == "Overview & Upload": _overview(dataframe, profile, quality)
    elif page == "Data Profiling": _profiling(profile)
    elif page == "Data Validation": _validation(dataframe)
    elif page == "Anomaly Detection": _anomalies(dataframe)
    elif page == "Quality Score": _quality(dataframe)
    else: _clean_and_report(dataframe, profile)
