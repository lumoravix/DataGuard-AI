"""Safe, compact HTML reporting for DataGuard AI."""

from __future__ import annotations

from html import escape


def _text(value) -> str:
    return escape(str(value), quote=True)


def _items(values: list[str]) -> str:
    return "".join(f"<li>{_text(value)}</li>" for value in values) or "<li>None</li>"


def generate_html_report(
    profile: dict,
    quality_score: dict,
    validation_results: list[dict],
    anomaly_results: dict | None = None,
    cleaning_result: dict | None = None,
) -> str:
    """Create an escaped HTML summary without including uploaded row-level data."""
    missing_rows = "".join(
        f"<tr><td>{_text(column)}</td><td>{_text(count)}</td><td>{_text(profile['missing_percentages'].get(column, 0))}%</td></tr>"
        for column, count in profile["missing_values"].items()
    ) or "<tr><td colspan='3'>No columns</td></tr>"
    validation_rows = "".join(
        "<tr>"
        f"<td>{_text(issue['issue_category'])}</td><td>{_text(issue.get('column') or '—')}</td>"
        f"<td>{_text(issue['affected_rows'])}</td><td>{_text(issue['explanation'])}</td>"
        "</tr>"
        for issue in validation_results
    ) or "<tr><td colspan='4'>No validation issues detected.</td></tr>"

    if quality_score["status"] == "success":
        components = quality_score["components"]
        quality_html = (
            f"<p>Overall score: <strong>{_text(quality_score['overall_score'])}/100</strong></p>"
            f"<p>Completeness: {_text(components['completeness'])}%; "
            f"Uniqueness: {_text(components['uniqueness'])}%; "
            f"Validity: {_text(components['validity'] if components['validity'] is not None else 'Not evaluated')}</p>"
            f"<ul>{_items(quality_score['explanations'])}</ul>"
        )
    else:
        quality_html = f"<p>Quality score unavailable: {_text(' '.join(quality_score['explanations']))}</p>"

    if anomaly_results and anomaly_results.get("status") == "success":
        anomaly_html = (
            f"<p>Detected {_text(anomaly_results['total_anomalies'])} model anomaly signal(s) "
            f"({_text(anomaly_results['anomaly_percentage'])}%). Features used: "
            f"{_text(', '.join(anomaly_results['numerical_features_used']))}.</p>"
        )
    else:
        anomaly_html = "<p>Anomaly detection was not enabled or was unavailable.</p>"

    if cleaning_result:
        cleaning_html = (
            f"<p>Before/after missing values: {_text(cleaning_result['before_missing'])}/"
            f"{_text(cleaning_result['after_missing'])}; duplicate rows: "
            f"{_text(cleaning_result['before_duplicates'])}/{_text(cleaning_result['after_duplicates'])}; "
            f"rows removed: {_text(cleaning_result['rows_removed'])}.</p>"
            f"<ul>{_items(cleaning_result['operations'] + cleaning_result['notes'] + cleaning_result.get('data_type_changes', []))}</ul>"
        )
    else:
        cleaning_html = "<p>No cleaned preview was generated; all metrics above describe the original upload.</p>"

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>DataGuard AI Report</title>
<style>body{{font-family:Arial,sans-serif;max-width:900px;margin:32px auto;color:#1f2937}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #d1d5db;padding:8px;text-align:left}}th{{background:#eff6ff}}h1,h2{{color:#1d4ed8}}</style></head>
<body><h1>DataGuard AI — Data Quality Report</h1>
<h2>Dataset overview</h2><p>Rows: {_text(profile['rows'])}; Columns: {_text(profile['columns'])}.</p>
<h2>Missing-value statistics</h2><table><tr><th>Column</th><th>Missing</th><th>Missing (%)</th></tr>{missing_rows}</table>
<h2>Duplicate-row statistics</h2><p>Additional duplicate rows: {_text(profile['duplicate_rows'])}.</p>
<h2>Data quality score</h2>{quality_html}
<h2>Configured validation issues</h2><table><tr><th>Category</th><th>Column</th><th>Affected rows</th><th>Explanation</th></tr>{validation_rows}</table>
<h2>Anomaly detection</h2>{anomaly_html}<p><em>Anomaly predictions are not confirmed data errors.</em></p>
<h2>Cleaning preview</h2>{cleaning_html}
</body></html>"""
