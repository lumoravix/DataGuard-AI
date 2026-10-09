"""DataGuard AI Streamlit entry point for CSV profiling."""

# The dashboard is isolated from the data-processing modules so visual changes
# cannot alter profiling, validation, scoring, cleaning, or ML behaviour.
import streamlit as _entry_streamlit
from dashboard import run_dashboard

run_dashboard()
_entry_streamlit.stop()

import hashlib

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.csv_loader import CSVLoadError, DEFAULT_MAX_FILE_SIZE_MB, load_csv
from src.profiler import missing_values_table, profile_dataframe
from src.validator import validate_dataframe
from src.anomaly_detector import DEFAULT_CONTAMINATION, detect_anomalies
from src.quality_score import calculate_quality_score
from src.cleaner import clean_dataframe, dataframe_to_csv_bytes
from src.report_generator import generate_html_report

st.set_page_config(page_title="DataGuard AI", page_icon="🛡️", layout="wide")
st.title("DataGuard AI — Intelligent Data Quality & Anomaly Detection Platform")
st.caption("Upload a CSV to explore its structure and data-quality profile.")

uploaded_file = st.file_uploader(
    "Upload a CSV file", type=["csv"],
    help=f"CSV files only, up to {DEFAULT_MAX_FILE_SIZE_MB} MB.",
)
if uploaded_file is None:
    st.info("Choose a CSV file to begin profiling.")
else:
    try:
        dataframe = load_csv(uploaded_file)
    except CSVLoadError as error:
        st.error(str(error))
    else:
        profile = profile_dataframe(dataframe)
        st.subheader("Dataset preview")
        st.dataframe(dataframe.head(10), use_container_width=True)

        metrics = st.columns(4)
        metrics[0].metric("Rows", profile["rows"])
        metrics[1].metric("Columns", profile["columns"])
        metrics[2].metric("Duplicate rows", profile["duplicate_rows"])
        metrics[3].metric("Missing values", sum(profile["missing_values"].values()))

        missing_table = missing_values_table(profile)
        st.subheader("Missing values by column")
        if missing_table["Missing values"].sum() == 0:
            st.success("No missing values found in this dataset.")
        else:
            chart = px.bar(missing_table, x="Column", y="Missing values", color="Missing (%)", color_continuous_scale="Blues")
            st.plotly_chart(chart, use_container_width=True)
        st.dataframe(missing_table, use_container_width=True, hide_index=True)

        st.subheader("Column details")
        column_details = missing_table.copy()
        column_details.insert(1, "Data type", [profile["data_types"][name] for name in column_details["Column"]])
        st.dataframe(column_details, use_container_width=True, hide_index=True)

        st.subheader("Numerical summary")
        if profile["numerical_summary"].empty:
            st.info("This dataset has no numerical columns to summarize.")
        else:
            st.dataframe(profile["numerical_summary"], use_container_width=True)

        st.divider()
        st.header("Data Quality Validation")
        st.caption("Validation identifies potential issues only; your uploaded data is never changed.")
        numeric_ranges = {}
        expected_types = {}
        with st.expander("Optional validation rules", expanded=False):
            numeric_columns = dataframe.select_dtypes(include="number").columns.tolist()
            if numeric_columns:
                range_column = st.selectbox("Numerical column to validate", numeric_columns)
                use_minimum = st.checkbox("Set a minimum value")
                minimum = st.number_input("Minimum", value=0.0) if use_minimum else None
                use_maximum = st.checkbox("Set a maximum value")
                maximum = st.number_input("Maximum", value=0.0) if use_maximum else None
                if use_minimum or use_maximum:
                    numeric_ranges[range_column] = {"min": minimum, "max": maximum}
            else:
                st.info("No numerical columns are available for range validation.")

            type_column = st.selectbox("Column with an expected type", dataframe.columns.tolist() or ["No columns"])
            expected_type = st.selectbox("Expected type", ["No type rule", "number", "text", "datetime", "boolean"])
            if expected_type != "No type rule" and len(dataframe.columns):
                expected_types[type_column] = expected_type

        validation_results = validate_dataframe(dataframe, numeric_ranges, expected_types)
        if validation_results:
            st.warning(f"Found {len(validation_results)} data-quality issue type(s).")
            validation_table = []
            for issue in validation_results:
                display_issue = issue.copy()
                display_issue["affected_row_indices"] = ", ".join(map(str, issue["affected_row_indices"]))
                validation_table.append(display_issue)
            st.dataframe(validation_table, use_container_width=True, hide_index=True)
        else:
            st.success("No data-quality issues were detected with the current rules.")

        st.divider()
        st.header("Data Quality Score")
        quality_score = calculate_quality_score(
            dataframe, numeric_ranges, expected_types, validation_results
        )
        if quality_score["status"] != "success":
            st.info(quality_score["explanations"][0])
        else:
            components = quality_score["components"]
            score_metrics = st.columns(4)
            score_metrics[0].metric("Overall score", f'{quality_score["overall_score"]}/100')
            score_metrics[1].metric("Completeness", f'{components["completeness"]}%')
            score_metrics[2].metric("Uniqueness", f'{components["uniqueness"]}%')
            score_metrics[3].metric("Validity", "Not evaluated" if components["validity"] is None else f'{components["validity"]}%')

            gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=quality_score["overall_score"],
                title={"text": "Overall data quality"},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#1f77b4"}},
            ))
            st.plotly_chart(gauge, use_container_width=True)
            evaluated_components = {
                name.title(): value for name, value in components.items() if value is not None
            }
            component_chart = px.bar(
                x=list(evaluated_components), y=list(evaluated_components.values()),
                range_y=[0, 100], labels={"x": "Component", "y": "Score (%)"},
                title="Evaluated quality components",
            )
            st.plotly_chart(component_chart, use_container_width=True)
            st.caption(
                "Effective weights — " + ", ".join(
                    f"{name.title()}: {weight}%" for name, weight in quality_score["effective_weights"].items()
                )
            )
            for explanation in quality_score["explanations"]:
                st.write(f"- {explanation}")
            if quality_score["validity_issue_explanations"]:
                st.warning("Validity reductions: " + " ".join(quality_score["validity_issue_explanations"]))

        st.divider()
        st.header("Anomaly Detection")
        st.caption("Unusual records are model signals, not confirmed data errors.")
        anomaly_results = None
        run_anomaly_detection = st.checkbox("Enable anomaly detection", value=False)
        if run_anomaly_detection:
            contamination = st.slider(
                "Expected proportion of unusual records",
                min_value=0.01,
                max_value=0.20,
                value=DEFAULT_CONTAMINATION,
                step=0.01,
                help="This is the model's expected anomaly rate, not a judgment about data correctness.",
            )
            anomaly_results = detect_anomalies(dataframe, contamination)
            if anomaly_results["status"] != "success":
                st.info(anomaly_results["message"])
            else:
                anomaly_metrics = st.columns(2)
                anomaly_metrics[0].metric("Anomalies detected", anomaly_results["total_anomalies"])
                anomaly_metrics[1].metric("Anomalous records", f'{anomaly_results["anomaly_percentage"]}%')
                st.caption(f'Numerical features used: {", ".join(anomaly_results["numerical_features_used"])}')
                st.info(anomaly_results["message"])

                score_chart = px.scatter(
                    anomaly_results["row_results"], x="row_index", y="anomaly_score",
                    color="prediction", color_discrete_map={"normal": "#1f77b4", "anomaly": "#d62728"},
                    title="Anomaly scores by row (lower scores are more unusual)",
                )
                st.plotly_chart(score_chart, use_container_width=True)

                flagged_mask = anomaly_results["row_results"]["prediction"].eq("anomaly").to_numpy()
                if flagged_mask.any():
                    st.subheader("Flagged records")
                    flagged_records = dataframe.iloc[flagged_mask].copy()
                    flagged_records.insert(0, "Row index", dataframe.index[flagged_mask])
                    flagged_records["Anomaly score"] = anomaly_results["row_results"].loc[flagged_mask, "anomaly_score"].to_numpy()
                    st.dataframe(flagged_records, use_container_width=True)
                else:
                    st.success("No records were flagged by the model.")

        st.divider()
        st.header("Data Cleaning")
        st.caption("Cleaning is optional. Imputation is a user-selected transformation, not a guaranteed correction.")
        numerical_cleaning = st.selectbox("Missing numerical values", ["unchanged", "mean", "median"])
        categorical_cleaning = st.selectbox("Missing categorical values", ["unchanged", "mode"])
        remove_duplicate_rows = st.checkbox("Remove duplicate rows (keep first occurrence)")
        st.subheader("Original dataset preview")
        st.dataframe(dataframe.head(10), use_container_width=True)
        cleaning_signature = (
            hashlib.sha256(uploaded_file.getvalue()).hexdigest(), numerical_cleaning,
            categorical_cleaning, remove_duplicate_rows,
        )
        if st.button("Generate cleaned preview"):
            st.session_state["cleaning_result"] = clean_dataframe(
                dataframe, numerical_cleaning, categorical_cleaning, remove_duplicate_rows
            )
            st.session_state["cleaning_signature"] = cleaning_signature
        cleaning_result = (
            st.session_state.get("cleaning_result")
            if st.session_state.get("cleaning_signature") == cleaning_signature
            else None
        )
        if cleaning_result:
            st.subheader("Cleaned dataset preview")
            st.dataframe(cleaning_result["cleaned_dataframe"].head(10), use_container_width=True)
            cleaning_metrics = st.columns(3)
            cleaning_metrics[0].metric("Missing values", f'{cleaning_result["before_missing"]} → {cleaning_result["after_missing"]}')
            cleaning_metrics[1].metric("Duplicate rows", f'{cleaning_result["before_duplicates"]} → {cleaning_result["after_duplicates"]}')
            cleaning_metrics[2].metric("Rows removed", cleaning_result["rows_removed"])
            st.write("Cleaning summary:")
            for operation in cleaning_result["operations"] + cleaning_result["notes"] + cleaning_result["data_type_changes"]:
                st.write(f"- {operation}")
            st.download_button(
                "Download Cleaned CSV", dataframe_to_csv_bytes(cleaning_result["cleaned_dataframe"]),
                file_name="dataguard_cleaned.csv", mime="text/csv",
            )

        st.divider()
        st.header("Data Quality Report")
        report_html = generate_html_report(
            profile, quality_score, validation_results, anomaly_results, cleaning_result
        )
        st.download_button(
            "Download Data Quality Report", report_html.encode("utf-8"),
            file_name="dataguard_data_quality_report.html", mime="text/html",
        )
