"""Streamlit UI for the self-evolving multi-agent analytics framework.

Run from the repository root with:
    python -m streamlit run dashboard.py

The dashboard supports both focused agent workflows and an explicitly
orchestrated sequential workflow from cleaning through explainable reporting.
"""

from __future__ import annotations

import hashlib
import html
import json
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from app.orchestrator.orchestrator import Orchestrator
from app.knowledge_graph.graph import KnowledgeGraph


PROJECT_ROOT = Path(__file__).resolve().parent
SAMPLE_DATASET = PROJECT_ROOT / "data" / "titanic.csv"
TEMP_ROOT = Path(tempfile.gettempdir()) / "self_evolving_multi_agent_demo"
UPLOAD_ROOT = TEMP_ROOT / "uploads"
GRAPH_ROOT = TEMP_ROOT / "graphs"
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
GRAPH_ROOT.mkdir(parents=True, exist_ok=True)


FULL_WORKFLOW_ACTION = "Full sequential analytics workflow"
SKIP_TARGET_LABEL = "— Skip ML and explainability —"


ACTIONS = {
    FULL_WORKFLOW_ACTION: {
        "agent": "SequentialWorkflow",
        "request": "run full sequential analytics workflow",
        "needs_target": False,
        "optional_target": True,
        "description": (
            "Run data cleaning → EDA → optional classification and fitted-model "
            "explainability → final report in one coordinated flow."
        ),
    },
    "Exploratory data analysis (EDA)": {
        "agent": "EDAAgent",
        "request": "Explore correlations and patterns in this dataset",
        "needs_target": False,
        "description": "Explore correlations, missingness, outliers, and ranked insights.",
    },
    "Data quality profile": {
        "agent": "DataAgent",
        "request": "Inspect the dataset for missing values",
        "needs_target": False,
        "description": "Profile dataset dimensions, missing values, and duplicate rows.",
    },
    "Train classification model": {
        "agent": "MLAgent",
        "request": "Predict the target using a classification model",
        "needs_target": True,
        "description": "Train and evaluate the classification model on held-out data.",
    },
    "Explain feature importance": {
        "agent": "ExplainabilityAgent",
        "request": "Explain feature importance for a classification model",
        "needs_target": True,
        "description": "Fit a reference model and compute held-out permutation importance.",
    },
    "Generate report from stored findings": {
        "agent": "ReportAgent",
        "request": "Generate a report explaining the dataset findings",
        "needs_target": False,
        "description": "Summarize findings and evidence already stored in the knowledge graph.",
    },
}


st.set_page_config(
    page_title="NEXUS AI | Explainable Analytics",
    page_icon="🔷",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
      :root {
        --nexus-ink: #142238;
        --nexus-muted: #687891;
        --nexus-line: #dce5ef;
        --nexus-paper: #f3f7fb;
        --nexus-teal: #087f8c;
        --nexus-mint: #69e0c2;
        --nexus-navy: #101827;
      }
      .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"],
      [data-testid="stMainBlockContainer"] {
        background: var(--nexus-paper);
        color: var(--nexus-ink);
      }
      [data-testid="stMainBlockContainer"] { padding-top: 1.4rem; max-width: 1560px; }
      [data-testid="stMain"] [data-testid="stMarkdownContainer"],
      [data-testid="stMain"] [data-testid="stMarkdownContainer"] h1,
      [data-testid="stMain"] [data-testid="stMarkdownContainer"] h2,
      [data-testid="stMain"] [data-testid="stMarkdownContainer"] h3,
      [data-testid="stMain"] [data-testid="stMarkdownContainer"] h4,
      [data-testid="stMain"] [data-testid="stExpander"] summary,
      [data-testid="stMain"] [data-testid="stTabs"] button,
      [data-testid="stMain"] label { color: var(--nexus-ink); }
      [data-testid="stMain"] [data-testid="stCaptionContainer"] { color: var(--nexus-muted); }
      [data-testid="stSidebar"] { background: #101827; border-right: 1px solid #24344a; }
      [data-testid="stSidebar"] * { color: #e8eef7; }
      [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] { background: #172337; border: 1px dashed #4b667c; }
      .hero {
        position: relative; overflow: hidden; padding: 1.7rem 1.9rem;
        border-radius: 24px;
        background: radial-gradient(circle at 92% 20%, rgba(105,224,194,.24), transparent 25%),
                    linear-gradient(118deg, #101827 0%, #1e3450 58%, #126d78 100%);
        color: #fff; margin-bottom: 1.1rem; border: 1px solid rgba(255,255,255,.08);
        box-shadow: 0 18px 50px rgba(20,34,56,.12);
      }
      .hero-kicker { color: #76e5d5; font-size: .76rem; font-weight: 800; letter-spacing: .16em; text-transform: uppercase; }
      .hero h1 { color: #fff; font-size: clamp(2rem, 3vw, 2.65rem); margin: .35rem 0 .5rem 0; letter-spacing: -.04em; }
      .hero p { color: #d8e3f1; margin: 0; max-width: 980px; line-height: 1.55; }
      .hero-chips { display:flex; flex-wrap:wrap; gap:.45rem; margin-top:1.05rem; }
      .hero-chip { display:inline-block; border:1px solid rgba(201,226,241,.24); color:#eaf6fb; background:rgba(255,255,255,.07); padding:.28rem .63rem; border-radius:999px; font-size:.76rem; }
      .section-label { color: var(--nexus-teal); font-size: .74rem; font-weight: 850; text-transform: uppercase; letter-spacing: .14em; margin:.2rem 0 .35rem 0; }
      .section-subtitle { color: var(--nexus-muted); font-size: .92rem; margin-top: -.25rem; margin-bottom: .85rem; }
      .mini-card { background:#fff; border:1px solid var(--nexus-line); border-radius:16px; padding:1rem 1.05rem; min-height:100px; box-shadow:0 5px 20px rgba(25,42,70,.035); }
      .mini-card-label { color:var(--nexus-muted); font-size:.76rem; font-weight:650; }
      .mini-card-value { color:var(--nexus-ink); font-size:1.55rem; font-weight:800; margin-top:.25rem; letter-spacing:-.035em; }
      .mini-card-note { color:var(--nexus-muted); font-size:.75rem; margin-top:.25rem; }
      div[data-testid="stMetric"] { background:#fff; border:1px solid var(--nexus-line); padding:.88rem 1rem; border-radius:15px; box-shadow:0 5px 20px rgba(25,42,70,.035); }
      div[data-testid="stMetricLabel"] { color:var(--nexus-muted); }
      div[data-testid="stMetricValue"] { color:var(--nexus-ink); font-weight:800; }
      [data-testid="stForm"] { background:#fff; border:1px solid var(--nexus-line); padding:1rem 1.1rem .7rem 1.1rem; border-radius:18px; box-shadow:0 5px 20px rgba(25,42,70,.035); }
      [data-testid="stTabs"] [role="tablist"] { gap:.25rem; border-bottom:1px solid var(--nexus-line); }
      [data-testid="stTabs"] button[role="tab"] { background:transparent; border-radius:10px 10px 0 0; padding:.7rem .85rem; font-weight:650; }
      [data-testid="stTabs"] button[aria-selected="true"] { background:#e2f5f2; color:#075d68 !important; }
      [data-testid="stExpander"] { background:#fff; border:1px solid var(--nexus-line); border-radius:13px; }
      [data-testid="stDataFrame"] { border:1px solid var(--nexus-line); border-radius:12px; overflow:hidden; }
      [data-testid="stButton"] button[kind="primary"], [data-testid="stFormSubmitButton"] button[kind="primary"] { background:linear-gradient(105deg,#087f8c,#0d6576); border:0; border-radius:11px; font-weight:750; min-height:2.75rem; box-shadow:0 8px 18px rgba(8,127,140,.18); }
      [data-testid="stButton"] button, [data-testid="stDownloadButton"] button { border-radius:10px; }
      .flow-caption { font-size:.78rem; color:var(--nexus-muted); margin:.25rem 0 .8rem 0; }
      .nexus-callout { padding:.8rem 1rem; background:#e8f4f7; border-left:4px solid #087f8c; border-radius:10px; color:#183c4c; margin:.4rem 0 .8rem 0; }
      .nexus-soft { color:var(--nexus-muted); font-size:.86rem; }
      footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _safe_text(value: Any, limit: int = 180) -> str:
    """Turn a value into a concise display string."""
    text = str(value if value is not None else "")
    text = " ".join(text.split())
    if len(text) > limit:
        return text[: limit - 1] + "…"
    return text


def _dot_escape(value: Any) -> str:
    return (
        _safe_text(value, 70)
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("<", "\\<")
        .replace(">", "\\>")
        .replace("\n", " ")
    )


def _graph_dot(
    graph: Any,
    current_finding_ids: set[str] | None = None,
    focus_node: str | None = None,
) -> str:
    """Build a bounded Graphviz view of current provenance neighbours."""
    if graph is None or graph.number_of_nodes() == 0:
        return 'digraph KnowledgeGraph { label="No graph data yet"; }'

    current_finding_ids = current_finding_ids or set()
    if focus_node in graph:
        visible = {focus_node}
    else:
        visible = {node_id for node_id in current_finding_ids if node_id in graph}

    for node_id in list(visible):
        visible.update(graph.predecessors(node_id))
        visible.update(graph.successors(node_id))
        for parent_id in list(graph.predecessors(node_id)):
            visible.update(graph.predecessors(parent_id))

    if not visible:
        visible = set(list(graph.nodes)[-30:])

    node_order = list(graph.nodes)
    if len(visible) > 42:
        keep = set(current_finding_ids)
        if focus_node in graph:
            keep.add(focus_node)
        for node_id in reversed(node_order):
            if len(visible) <= 42:
                break
            if node_id not in keep:
                visible.discard(node_id)

    lines = [
        "digraph KnowledgeGraph {",
        '  graph [rankdir=LR, bgcolor="transparent", pad="0.2", nodesep="0.35", ranksep="0.65"];',
        '  node [style="rounded,filled", fontname="Arial", fontsize="10", color="#d9e2ef", fontcolor="#172033", margin="0.12,0.08"];',
        '  edge [fontname="Arial", fontsize="8", color="#8292a8", fontcolor="#526176", arrowsize="0.7"];',
    ]

    shapes = {
        "dataset": ("folder", "#dbeafe"),
        "feature": ("ellipse", "#e0e7ff"),
        "finding": ("box", "#ccfbf1"),
        "evidence": ("note", "#fef3c7"),
        "model": ("component", "#f3e8ff"),
        "execution": ("diamond", "#fee2e2"),
    }

    for node_id in node_order:
        if node_id not in visible:
            continue
        attrs = graph.nodes[node_id]
        node_type = _safe_text(attrs.get("type", "node"), 30).lower()
        shape, fill = shapes.get(node_type, ("ellipse", "#f1f5f9"))
        title = (
            attrs.get("description")
            or attrs.get("name")
            or attrs.get("dataset")
            or attrs.get("task")
            or node_id
        )
        label = f"{node_type.title()}\\n{_dot_escape(title)}"
        lines.append(
            f'  "{_dot_escape(node_id)}" [label="{label}", shape="{shape}", fillcolor="{fill}"];'
        )

    for source, target, edge_attrs in graph.edges(data=True):
        if source not in visible or target not in visible:
            continue
        relation = _dot_escape(edge_attrs.get("relation", "related_to"))
        lines.append(
            f'  "{_dot_escape(source)}" -> "{_dot_escape(target)}" [label="{relation}"];'
        )

    lines.append("}")
    return "\n".join(lines)


def _get_graph_columns(dataset_path: Path) -> list[str]:
    """Read the CSV header only, without loading the entire dataset."""
    if not dataset_path.exists():
        return []
    try:
        return [str(column) for column in pd.read_csv(dataset_path, nrows=0).columns]
    except (OSError, ValueError, pd.errors.ParserError, UnicodeDecodeError):
        return []


@st.cache_data(show_spinner=False)
def _load_csv_preview(path_text: str, modified_ns: int, file_size: int, nrows: int = 10000) -> pd.DataFrame:
    """Load a bounded CSV preview; file metadata invalidates the cache on update."""
    del modified_ns, file_size  # These arguments intentionally participate in the cache key.
    return pd.read_csv(path_text, nrows=nrows, low_memory=False)


def _dataset_file_size(path: Path) -> str:
    try:
        size = path.stat().st_size
    except OSError:
        return "Unavailable"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.2f} MB"


def _read_preview(path: Path, nrows: int = 10000) -> tuple[pd.DataFrame, str | None]:
    if not path.exists():
        return pd.DataFrame(), f"File not found: {path}"
    try:
        stat = path.stat()
        return _load_csv_preview(str(path), stat.st_mtime_ns, stat.st_size, nrows), None
    except (OSError, ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        return pd.DataFrame(), f"Could not preview CSV: {exc}"


def _result_agent(item: dict[str, Any]) -> str:
    details = item.get("analysis_result") or {}
    return str(item.get("agent") or details.get("agent") or "")


def _workflow_stages(
    action_label: str,
    output: dict[str, Any] | None,
    dataset_exists: bool,
) -> list[dict[str, str]]:
    """Derive workflow labels from actual recorded output; never fake stage completion."""
    stages = [
        {"key": "input", "label": "CSV input", "detail": "File available and readable"},
        {"key": "cleaning", "label": "Data cleaning", "detail": "Normalize schema and record safe changes"},
        {"key": "eda", "label": "EDA", "detail": "Analyze missingness, correlations, and outliers"},
        {"key": "ml", "label": "Machine learning", "detail": "Train and evaluate when a target is selected"},
        {"key": "xai", "label": "Explainability", "detail": "Explain the fitted model when ML runs"},
        {"key": "verify", "label": "Evidence verification", "detail": "Check structure and provenance"},
        {"key": "report", "label": "Final report", "detail": "Summarize stored findings and evidence"},
    ]
    if action_label == "Data quality profile":
        stages[1]["label"] = "Data quality profile"
        stages[1]["detail"] = "Inspect dataset dimensions, missing values, and duplicates"
    for stage in stages:
        stage["state"] = "Waiting for run"

    stages[0]["state"] = "Ready" if dataset_exists else "Missing file"
    if output is None:
        if action_label == FULL_WORKFLOW_ACTION:
            stages[1]["state"] = "Ready to run"
            stages[2]["state"] = "Ready to run"
            stages[3]["state"] = "Conditional"
            stages[4]["state"] = "Conditional"
            stages[5]["state"] = "Ready to run"
            stages[6]["state"] = "Ready to run"
        else:
            focus = {
                "Exploratory data analysis (EDA)": "eda",
                "Data quality profile": "cleaning",
                "Train classification model": "ml",
                "Explain feature importance": "xai",
                "Generate report from stored findings": "report",
            }.get(action_label)
            for stage in stages[1:]:
                stage["state"] = "Selected" if stage["key"] == focus else "Not in focused run"
        return stages

    results = output.get("results", [])
    errors = output.get("errors", [])
    agents = {_result_agent(item) for item in results}
    task_names = [str(item.get("task") or item.get("task_id") or "").lower() for item in results]
    checks = [check for item in results for check in item.get("verification", [])]
    errors_text = " ".join(str(error) for error in errors).lower()
    is_full = action_label == FULL_WORKFLOW_ACTION

    stages[0]["state"] = "Complete" if dataset_exists else "Missing file"
    def stage_state(agent: str, task_token: str, error_token: str | None = None) -> str:
        matched = agent in agents or any(task_token in name for name in task_names)
        if matched:
            return "Complete"
        if error_token and (error_token.lower() in errors_text):
            return "Needs attention"
        return "Pending"

    if is_full:
        stages[1]["state"] = stage_state("DataCleaningAgent", "stage 1 — data cleaning", "datacleaningagent")
        stages[2]["state"] = stage_state("EDAAgent", "stage 2 — eda", "edaagent")
        if any("prediction_skipped" in str(item.get("task_id", "")) or "predictive modeling skipped" in str(item.get("task", "")).lower() for item in results):
            stages[3]["state"] = "Skipped — no target"
            stages[4]["state"] = "Skipped — no model"
        else:
            stages[3]["state"] = stage_state("MLAgent", "stage 3 — ml", "mlagent")
            stages[4]["state"] = stage_state("ExplainabilityAgent", "stage 4 — explainability", "explainabilityagent")
        stages[6]["state"] = stage_state("ReportAgent", "stage 5 — final report", "reportagent")
    else:
        focused_key = {
            "Exploratory data analysis (EDA)": "eda",
            "Data quality profile": "cleaning",
            "Train classification model": "ml",
            "Explain feature importance": "xai",
            "Generate report from stored findings": "report",
        }.get(action_label)
        for stage in stages[1:]:
            if stage["key"] == focused_key:
                agent = {"cleaning": "DataAgent", "eda": "EDAAgent", "ml": "MLAgent", "xai": "ExplainabilityAgent", "report": "ReportAgent"}.get(focused_key, "")
                stages[[s["key"] for s in stages].index(stage["key"])] ["state"] = "Complete" if agent in agents or any(focused_key in name for name in task_names) else "Needs attention"
            else:
                stage["state"] = "Not in focused run"

    if checks:
        stages[5]["state"] = "Verified" if all(check.get("structural_verified") is True for check in checks) else "Needs review"
    else:
        stages[5]["state"] = "Not recorded"

    # Promote failed overall workflow even when a partial task exists.
    if output.get("status") == "completed_with_errors":
        for stage in stages:
            if stage["state"] in {"Pending", "Needs attention"}:
                stage["state"] = "Needs attention"
    return stages


def _workflow_dot(stages: list[dict[str, str]]) -> str:
    """Return an execution-aware flow diagram using only recorded stage states."""
    colors = {
        "Complete": "#d1fae5", "Verified": "#d1fae5", "Ready": "#dbeafe",
        "Ready to run": "#e0f2fe", "Selected": "#dbeafe", "Conditional": "#fef3c7",
        "Skipped — no target": "#f1f5f9", "Skipped — no model": "#f1f5f9",
        "Not in focused run": "#f1f5f9", "Waiting for run": "#f8fafc",
        "Pending": "#f8fafc", "Needs attention": "#fee2e2", "Needs review": "#fee2e2",
        "Missing file": "#fee2e2", "Not recorded": "#fef3c7",
    }
    lines = [
        "digraph NEXUSWorkflow {",
        'graph [rankdir=LR, bgcolor="transparent", pad="0.22", nodesep="0.3", ranksep="0.45", splines=ortho];',
        'node [shape=box, style="rounded,filled", fontname="Arial", fontsize=10, margin="0.18,0.13", color="#d7e1ec", fontcolor="#142238", penwidth=1.1];',
        'edge [color="#91a4b8", penwidth=1.25, arrowsize=0.7];',
    ]
    for index, stage in enumerate(stages):
        title = _dot_escape(stage["label"])
        state = _dot_escape(stage["state"])
        color = colors.get(stage["state"], "#f8fafc")
        # Keep the Graphviz line-break escape outside _dot_escape so it stays a line break.
        lines.append(f'"{stage["key"]}" [label="{title}\\n{state}", fillcolor="{color}"];')
        if index:
            lines.append(f'"{stages[index-1]["key"]}" -> "{stage["key"]}";')
    lines.append("}")
    return "\n".join(lines)


def _strategy_decisions(output: dict[str, Any] | None) -> list[dict[str, Any]]:
    rows = []
    if not output:
        return rows
    for item in output.get("results", []):
        details = item.get("analysis_result") or {}
        if details.get("task") != "investigate_correlation" or "method" not in details:
            continue
        context = details.get("strategy_context") or {}
        features = context.get("features") or {}
        reason = str(details.get("strategy_reason") or "No rationale recorded.")
        if "Using historical strategy recommendation" in reason:
            source = "Contextual memory"
        elif "Contextual memory abstained" in reason:
            source = "Heuristic fallback after abstention"
        elif "explicitly specified" in reason.lower():
            source = "Explicit override"
        elif "skew" in reason.lower():
            source = "Data-based heuristic"
        else:
            source = "Recorded strategy"
        rows.append({
            "Column pair": " ↔ ".join(str(value) for value in details.get("columns", [])),
            "Selected method": details.get("method", "—"),
            "Decision source": source,
            "Complete rows": details.get("sample_size", context.get("complete_sample_size", "—")),
            "Missing %": round(float(features.get("missing_fraction", 0)) * 100, 1) if features else "—",
            "Max |skew|": round(float(features.get("max_abs_skew", 0)), 2) if features else "—",
            "Outlier %": round(float(features.get("max_iqr_outlier_fraction", 0)) * 100, 1) if features else "—",
            "|Pearson − Spearman|": round(float(features.get("pearson_spearman_gap", 0)), 3) if features else "—",
            "Rationale": reason,
        })
    return rows


def _node_label(graph: Any, node_id: str) -> str:
    attrs = graph.nodes[node_id]
    kind = str(attrs.get("type", "node")).title()
    label = attrs.get("description") or attrs.get("name") or attrs.get("dataset") or attrs.get("task") or node_id
    return f"{kind} · {_safe_text(label, 90)}"


def _get_evidence_description(graph: Any, evidence_id: str) -> str:
    if graph is None or evidence_id not in graph:
        return "Evidence details are unavailable in this graph snapshot."

    attrs = graph.nodes[evidence_id]
    description = attrs.get("description") or attrs.get("name") or evidence_id
    method = attrs.get("method")
    value = attrs.get("value")
    strategy_reason = attrs.get("strategy_reason")

    parts = [_safe_text(description, 400)]

    if method:
        parts.append(f"Method: {_safe_text(method, 120)}")

    if value is not None:
        parts.append(f"Value: {_safe_text(value, 180)}")

    if strategy_reason:
        parts.append(
            f"Strategy reason: {_safe_text(strategy_reason, 240)}"
        )

    return " | ".join(parts)

def _task_rows(output: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in output.get("results", []):
        checks = item.get("verification", [])
        evaluation = item.get("evaluation") or {}
        rows.append(
            {
                "Task": _safe_text(item.get("task") or item.get("task_id") or "Agent task", 90),
                "Findings": len(item.get("finding_ids", [])),
                "Structural checks passed": sum(
                    check.get("structural_verified") is True for check in checks
                ),
                "Structural checks": len(checks),
                "Execution marked successful": evaluation.get("successful", "Not evaluated"),
            }
        )
    return rows


def _render_agent_details(item: dict[str, Any]) -> None:
    """Render the domain-specific output for one agent result."""
    details = item.get("analysis_result") or {}

    if details.get("agent") == "ReportAgent" or "report_markdown" in details:
        report_text = details.get("report_markdown", "")
        st.markdown(report_text or "No Markdown report was returned.")
        if report_text:
            st.download_button(
                "Download Markdown report",
                data=report_text.encode("utf-8"),
                file_name="data_analytics_report.md",
                mime="text/markdown",
                key="download_report_markdown",
            )
        return

    if details.get("agent") == "ExplainabilityAgent" or "feature_importance" in details:
        st.markdown("#### Feature importance")
        metric_cols = st.columns(3)
        metric_cols[0].metric(
            "Held-out accuracy",
            f"{details.get('baseline_accuracy', 0):.3f}"
            if isinstance(details.get("baseline_accuracy"), (int, float))
            else "—",
        )
        metric_cols[1].metric(
            "Macro F1",
            f"{details.get('f1_macro', 0):.3f}"
            if isinstance(details.get("f1_macro"), (int, float))
            else "—",
        )
        metric_cols[2].metric(
            "Test samples",
            details.get("test_samples", "—"),
        )
        if details.get("model_source") == "exact_fitted_model_from_MLAgent":
            st.caption(
                "These permutation-importance scores explain the exact fitted model "
                "returned by MLAgent, evaluated on the same held-out test partition."
            )
        else:
            st.caption(
                "These importance scores describe a newly fitted reference model, "
                "not the exact model instance produced by MLAgent."
            )
        importance = details.get("feature_importance", [])
        if importance:
            st.dataframe(pd.DataFrame(importance), width="stretch", hide_index=True)
        else:
            st.info("No feature-importance rows were returned.")
        with st.expander("Explanation method and limitations"):
            st.write(f"Method: {details.get('explanation_method', 'Not specified')}")
            for limitation in details.get("limitations", []):
                st.markdown(f"- {limitation}")
        with st.expander("Full explanation result"):
            st.json(details)
        return

    if details.get("model_name") == "RandomForestClassifier" or "target_column" in details:
        st.markdown("#### Classification results")
        metric_cols = st.columns(3)
        metrics = details.get("metrics", {})
        metric_cols[0].metric("Accuracy", f"{metrics['accuracy']:.3f}" if isinstance(metrics.get("accuracy"), (int, float)) else "—")
        metric_cols[1].metric("Macro precision", f"{metrics['precision_macro']:.3f}" if isinstance(metrics.get("precision_macro"), (int, float)) else "—")
        metric_cols[2].metric("Macro F1", f"{metrics['f1_macro']:.3f}" if isinstance(metrics.get("f1_macro"), (int, float)) else "—")
        st.caption(f"Target column: {details.get('target_column', 'Not specified')}")
        with st.expander("Full classification result"):
            st.json(details)
        return

    st.markdown("#### Agent output")
    if details:
        st.json(details)
    else:
        st.info("This result did not return a separate analysis-result object.")


def _show_findings(output: dict[str, Any], graph: Any) -> None:
    results = output.get("results", [])
    if not results:
        st.info("No finding records were returned for this action.")
        return

    for item_index, item in enumerate(results):
        finding_ids = item.get("finding_ids", [])
        findings = item.get("findings", [])
        checks = item.get("verification", [])
        title = _safe_text(item.get("task") or item.get("task_id") or "Agent task", 100)
        st.markdown(f"#### {title}")

        if not finding_ids:
            if findings:
                for finding in findings:
                    st.markdown(f"- {_safe_text(finding, 500)}")
            else:
                st.caption("No knowledge-graph finding IDs were returned.")

        for index, finding_id in enumerate(finding_ids):
            finding_text = findings[index] if index < len(findings) else finding_id
            check = checks[index] if index < len(checks) else {}
            with st.expander(f"{_safe_text(finding_id, 65)} · {_safe_text(finding_text, 140)}"):
                st.write(finding_text)
                if check.get("structural_verified") is True:
                    st.success("Structural validation passed")
                elif check:
                    st.warning("One or more structural checks did not pass")
                else:
                    st.caption("No finding-level verification result was returned for this item.")

                st.info(
                    "The current FactChecker checks graph structure and metadata; "
                    "it does not establish scientific truth."
                )

                evidence_ids = check.get("evidence", [])
                if evidence_ids:
                    st.markdown("**Linked evidence**")
                    for evidence_id in evidence_ids:
                        st.markdown(
                            f"- **{_safe_text(evidence_id, 80)}:** "
                            f"{_get_evidence_description(graph, evidence_id)}"
                        )

                model_ids = check.get("models", [])
                if model_ids:
                    st.markdown("**Linked model metadata**")
                    for model_id in model_ids:
                        model_attrs = graph.nodes[model_id] if graph is not None and model_id in graph else {}
                        st.markdown(
                            f"- **{_safe_text(model_id, 80)}:** "
                            f"{_safe_text(model_attrs.get('model_name', 'Model'), 120)}"
                        )

                with st.expander("Structural checks (technical details)"):
                    st.json(check.get("checks", {}))

        evaluation = item.get("evaluation")
        if evaluation:
            with st.expander(f"Execution evaluation · {title}"):
                st.json(evaluation)


def _workflow_request(action_label: str) -> str:
    """Return a canonical router phrase for the chosen dashboard action."""
    return ACTIONS[action_label]["request"]


# Sidebar: dataset selection and active capability summary.
with st.sidebar:
    st.markdown("## 🔷 NEXUS AI")
    st.caption("Explainable multi-agent analytics")
    st.markdown("---")
    st.markdown("### Dataset")
    uploaded_file = st.file_uploader(
        "Upload a CSV file",
        type=["csv"],
        help="Leave empty to use the included Titanic example dataset.",
    )

    if uploaded_file is not None:
        upload_bytes = uploaded_file.getvalue()
        full_digest = hashlib.sha256(upload_bytes).hexdigest()
        safe_filename = Path(uploaded_file.name).name or "uploaded.csv"
        # Content-addressed paths keep the uploaded dataset identity stable
        # across Streamlit sessions, so its graph-backed history can be reused.
        dataset_dir = UPLOAD_ROOT / full_digest
        dataset_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = dataset_dir / safe_filename
        if not dataset_path.exists() or hashlib.sha256(dataset_path.read_bytes()).hexdigest() != full_digest:
            dataset_path.write_bytes(upload_bytes)
        dataset_label = safe_filename
        dataset_fingerprint = full_digest
    else:
        dataset_path = SAMPLE_DATASET
        dataset_label = "Titanic sample dataset"
        dataset_fingerprint = "sample_titanic"

    st.markdown(f"**Selected:** {_safe_text(dataset_label, 55)}")
    if not dataset_path.exists():
        st.error(f"Dataset not found: {_safe_text(dataset_path, 160)}")

    st.markdown("---")
    st.markdown("### Available agents")
    st.markdown(
        "- **DataCleaningAgent:** traceable CSV cleaning\n"
        "- **DataAgent:** data-quality profile\n"
        "- **EDAAgent:** correlations, missingness, outliers\n"
        "- **MLAgent:** classification\n"
        "- **ExplainabilityAgent:** permutation importance\n"
        "- **ReportAgent:** report from stored findings"
    )
    st.caption(
        "Choose the full sequential workflow to chain cleaning, EDA, optional ML, "
        "explainability of the fitted model, and a final report. Focused agent actions remain available below."
    )


column_names = _get_graph_columns(dataset_path)
graph_path = GRAPH_ROOT / f"{dataset_fingerprint}.json"


# Bind the visible run to the currently selected dataset graph. If the user
# switches datasets, do not present a stale result as if it came from the new CSV.
saved_output = st.session_state.get("last_output")
has_current_run = (
    saved_output is not None
    and st.session_state.get("last_graph_path") == str(graph_path)
)
output = saved_output if has_current_run else None
view_action = st.session_state.get("last_action", FULL_WORKFLOW_ACTION)
view_target = st.session_state.get("last_target_column") if has_current_run else None

if has_current_run and st.session_state.get("last_graph") is not None:
    graph_snapshot = st.session_state.last_graph
else:
    try:
        graph_snapshot = KnowledgeGraph(storage_path=graph_path).get_graph().copy()
    except (OSError, ValueError, json.JSONDecodeError):
        graph_snapshot = None

results = output.get("results", []) if output else []
errors = output.get("errors", []) if output else []

source_preview, source_preview_error = _read_preview(dataset_path)
if source_preview_error:
    st.warning(source_preview_error)

cleaning_result = next(
    (
        item for item in results
        if _result_agent(item) == "DataCleaningAgent"
        or "cleaned_csv_path" in (item.get("analysis_result") or {})
    ),
    None,
)
cleaning_details = (cleaning_result or {}).get("analysis_result") or {}
cleaned_path = Path(cleaning_details["cleaned_csv_path"]) if cleaning_details.get("cleaned_csv_path") else None
cleaned_path = cleaned_path if cleaned_path and cleaned_path.exists() else None

st.markdown('<div class="section-label">Workspace</div>', unsafe_allow_html=True)
hero_dataset = html.escape(_safe_text(dataset_label, 90))
hero_status = "Results available" if has_current_run else "Ready for analysis"
st.markdown(
    f"""
    <div class="hero">
      <div class="hero-kicker">NEXUS AI · SELF-EVOLVING ANALYTICS</div>
      <h1>Understand your data. Trace every decision.</h1>
      <p>Coordinate cleaning, exploratory analysis, optional machine learning, model explainability, and evidence-linked reporting in one workspace. Inspect the strategy selected, the evidence behind it, and the experience used to inform it.</p>
      <div class="hero-chips">
        <span class="hero-chip">Dataset · {hero_dataset}</span>
        <span class="hero-chip">{html.escape(hero_status)}</span>
        <span class="hero-chip">Knowledge graph · {'Loaded' if graph_snapshot is not None and graph_snapshot.number_of_nodes() else 'No stored nodes yet'}</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.form("analysis_form"):
    form_cols = st.columns([1.4, 1.0])
    with form_cols[0]:
        action_label = st.selectbox(
            "Choose a workflow",
            options=list(ACTIONS.keys()),
            index=list(ACTIONS.keys()).index(view_action) if view_action in ACTIONS else 0,
            help="Choose the full sequential pipeline or run one focused capability.",
        )
    with form_cols[1]:
        target_column = None
        requires_target = ACTIONS[action_label]["needs_target"]
        optional_target = ACTIONS[action_label].get("optional_target", False)
        if requires_target or optional_target:
            if column_names:
                if optional_target:
                    target_options = [SKIP_TARGET_LABEL] + column_names
                    if "Survived" in column_names:
                        default_target_index = 1 + column_names.index("Survived")
                    else:
                        default_target_index = 0
                    selected_target = st.selectbox(
                        "Prediction target (optional)",
                        options=target_options,
                        index=default_target_index,
                        help="Choose a target to run ML and explain the exact fitted model, or skip predictive stages.",
                    )
                    target_column = None if selected_target == SKIP_TARGET_LABEL else selected_target
                else:
                    default_target_index = column_names.index("Survived") if "Survived" in column_names else 0
                    target_column = st.selectbox(
                        "Target column",
                        options=column_names,
                        index=default_target_index,
                        help="This column is the label to predict and is excluded from model inputs.",
                    )
            else:
                st.warning("Upload a valid CSV before selecting a prediction target.")
        else:
            st.markdown("**Current mode**")
            st.write("Analysis and evidence capture")
    st.caption(ACTIONS[action_label]["description"])
    run_clicked = st.form_submit_button(
        "▶ Run selected workflow",
        type="primary",
        use_container_width=True,
    )

if run_clicked:
    st.session_state.last_error = None
    if not dataset_path.exists():
        st.session_state.last_error = "The selected dataset could not be found. Upload a CSV file and try again."
    elif ACTIONS[action_label]["needs_target"] and not target_column:
        st.session_state.last_error = "This workflow requires a target column. Choose a valid target column."
    else:
        spinner_label = (
            "Running sequential workflow: cleaning → EDA → optional ML/XAI → report…"
            if action_label == FULL_WORKFLOW_ACTION
            else f"Running {ACTIONS[action_label]['agent']} and updating the knowledge graph…"
        )
        try:
            with st.spinner(spinner_label):
                orchestrator = Orchestrator(dataset_path=str(dataset_path), graph_path=graph_path)
                run_output = orchestrator.process_request(
                    _workflow_request(action_label),
                    target_column=target_column,
                )
                st.session_state.last_output = run_output
                st.session_state.last_graph = orchestrator.knowledge_graph.get_graph().copy()
                st.session_state.last_dataset_label = dataset_label
                st.session_state.last_dataset_path = str(dataset_path)
                st.session_state.last_request = _workflow_request(action_label)
                st.session_state.last_action = action_label
                st.session_state.last_graph_path = str(graph_path)
                st.session_state.last_target_column = target_column
            st.rerun()
        except Exception as exc:
            st.session_state.last_error = f"{type(exc).__name__}: {exc}"

if st.session_state.get("last_error"):
    st.error("The workflow could not be completed. Review the technical details below.")
    with st.expander("Technical details"):
        st.code(st.session_state.last_error)

# Rebind after an execution so the view uses the just-finished output.
saved_output = st.session_state.get("last_output")
has_current_run = saved_output is not None and st.session_state.get("last_graph_path") == str(graph_path)
output = saved_output if has_current_run else None
view_action = st.session_state.get("last_action", action_label) if has_current_run else action_label
view_target = st.session_state.get("last_target_column") if has_current_run else target_column
results = output.get("results", []) if output else []
errors = output.get("errors", []) if output else []
if has_current_run and st.session_state.get("last_graph") is not None:
    graph_snapshot = st.session_state.last_graph

verification_checks = [check for item in results for check in item.get("verification", [])]
structural_count = sum(check.get("structural_verified") is True for check in verification_checks)
finding_count = sum(len(item.get("finding_ids", [])) for item in results)
execution_count = sum(1 for _, attrs in graph_snapshot.nodes(data=True) if attrs.get("type") == "execution") if graph_snapshot is not None else 0
node_count = graph_snapshot.number_of_nodes() if graph_snapshot is not None else 0
edge_count = graph_snapshot.number_of_edges() if graph_snapshot is not None else 0

st.markdown('<div class="section-label">At a glance</div>', unsafe_allow_html=True)
if output:
    status = output.get("status", "unknown")
    message = output.get("message", "")
    if status == "completed":
        st.success(message or "Workflow completed.")
    elif status == "completed_with_warnings":
        st.warning(message or "Workflow completed with warnings.")
    elif status == "completed_with_errors":
        st.error(message or "Workflow completed with errors.")
    elif status == "input_required":
        st.info(message or "Additional input is required.")
    elif status == "no_findings":
        st.info(message or "No stored findings were available for this report.")
    elif status == "not_implemented":
        st.warning(message or "This agent is not implemented.")
else:
    status = "ready"
    st.caption("Upload a CSV or use the Titanic sample, choose a workflow, and run it to populate the workspace.")

if output:
    metric_cols = st.columns(4)
    metric_cols[0].metric("Agent tasks", output.get("task_count", len(results)))
    metric_cols[1].metric("Findings returned", finding_count)
    metric_cols[2].metric("Structural checks passed", f"{structural_count}/{len(verification_checks)}")
    metric_cols[3].metric("Recorded executions", execution_count)
else:
    file_size = _dataset_file_size(dataset_path) if dataset_path.exists() else "Unavailable"
    metric_cols = st.columns(4)
    metric_cols[0].metric("Dataset columns", len(column_names))
    metric_cols[1].metric("Preview rows", len(source_preview))
    metric_cols[2].metric("CSV size", file_size)
    metric_cols[3].metric("Saved executions", execution_count)

st.caption(
    f"Dataset: {st.session_state.get('last_dataset_label', dataset_label) if has_current_run else dataset_label} · "
    f"Workflow: {view_action} · Target: {view_target or 'Not applicable'}"
)

# A single workspace keeps the visual story ordered: understand → inspect → explain → trace → report.
tab_overview, tab_workflow, tab_explorer, tab_visuals, tab_results, tab_findings, tab_graph, tab_memory, tab_reports = st.tabs(
    ["✦ Overview", "⇢ Workflow", "▦ Data explorer", "◫ Visual analytics", "🧠 Agent output", "⌕ Findings & XAI", "◎ Knowledge graph", "↻ Memory & history", "▤ Reports & evaluation"]
)

with tab_overview:
    st.markdown('<div class="section-label">Overview</div>', unsafe_allow_html=True)
    st.subheader("Your analytics workspace")
    st.markdown('<div class="section-subtitle">A single place to follow the data from input to evidence-backed report.</div>', unsafe_allow_html=True)
    summary_cols = st.columns(3)
    with summary_cols[0]:
        st.markdown('<div class="mini-card"><div class="mini-card-label">01 · Analyze</div><div class="mini-card-value">Profile & discover</div><div class="mini-card-note">Inspect quality, distributions, correlations, and outliers.</div></div>', unsafe_allow_html=True)
    with summary_cols[1]:
        st.markdown('<div class="mini-card"><div class="mini-card-label">02 · Explain</div><div class="mini-card-value">Trace decisions</div><div class="mini-card-note">Review selected methods, model explanations, and linked evidence.</div></div>', unsafe_allow_html=True)
    with summary_cols[2]:
        st.markdown('<div class="mini-card"><div class="mini-card-label">03 · Remember</div><div class="mini-card-value">Reuse experience</div><div class="mini-card-note">Inspect contextual memory and prior recorded strategy decisions.</div></div>', unsafe_allow_html=True)
    st.markdown("### Latest run summary")
    if not output:
        st.info("No run has been recorded for this dataset in the current view. The data explorer and planned workflow are available before execution.")
    else:
        st.write(f"**Request:** {output.get('user_request', '')}")
        st.write(f"**Selected component:** {output.get('selected_agent', 'Unknown')}")
        st.write(f"**Status:** {status}")
        if results:
            st.dataframe(pd.DataFrame(_task_rows(output)), width="stretch", hide_index=True)
        if errors:
            with st.expander(f"Stage warnings/errors ({len(errors)})"):
                st.json(errors)
    st.markdown("### Core capabilities")
    cap_cols = st.columns(4)
    cards = [
        ("Data quality", "Missing values, duplicate rows, schema issues"),
        ("Adaptive analysis", "Pearson/Spearman and profile-based memory"),
        ("Explainability", "Feature-importance evidence for fitted models"),
        ("Traceability", "Findings, evidence, execution and graph history"),
    ]
    for col, (title, desc) in zip(cap_cols, cards):
        with col:
            st.markdown(f'<div class="mini-card"><div class="mini-card-label">NEXUS AI</div><div style="font-weight:800;font-size:1.02rem;margin:.35rem 0;">{html.escape(title)}</div><div class="mini-card-note">{html.escape(desc)}</div></div>', unsafe_allow_html=True)

with tab_workflow:
    st.markdown('<div class="section-label">Orchestration</div>', unsafe_allow_html=True)
    st.subheader("CSV → coordinated agent workflow")
    st.markdown('<div class="section-subtitle">Stage states are derived from the latest recorded run. Conditional stages are skipped when no prediction target is selected.</div>', unsafe_allow_html=True)
    stages = _workflow_stages(view_action, output, dataset_path.exists())
    try:
        st.graphviz_chart(_workflow_dot(stages), width="stretch")
    except Exception as exc:
        st.warning("The flowchart renderer is unavailable; showing the same stage states as a table.")
        st.caption(_safe_text(exc, 250))
    st.dataframe(pd.DataFrame([{"Stage": x["label"], "Status": x["state"], "Responsibility": x["detail"]} for x in stages]), width="stretch", hide_index=True)
    if view_action == FULL_WORKFLOW_ACTION:
        st.markdown('<div class="nexus-callout">Full flow: cleaning → EDA → optional ML → fitted-model explanation → evidence verification → report. A stage is marked complete only when the returned execution record supports that status.</div>', unsafe_allow_html=True)
    else:
        st.info("You selected a focused workflow. Other stages are marked as outside this run rather than falsely shown as completed.")

with tab_explorer:
    st.markdown('<div class="section-label">Dataset explorer</div>', unsafe_allow_html=True)
    st.subheader("Inspect the CSV before trusting the results")
    inspect_options = {"Source CSV": dataset_path}
    if cleaned_path is not None:
        inspect_options["Cleaned analysis copy"] = cleaned_path
    inspect_choice = st.radio("Dataset version", list(inspect_options.keys()), horizontal=True, key="dataset_explorer_source")
    inspect_path = inspect_options[inspect_choice]
    preview_df, preview_error = _read_preview(inspect_path)
    if preview_error:
        st.error(preview_error)
    elif preview_df.empty:
        st.warning("No preview rows are available for this CSV.")
    else:
        missing_cells = int(preview_df.isna().sum().sum())
        numeric_cols = preview_df.select_dtypes(include="number").columns.tolist()
        categorical_cols = [str(col) for col in preview_df.columns if col not in numeric_cols]
        pcols = st.columns(4)
        pcols[0].metric("Rows in preview", f"{len(preview_df):,}", help="Preview is capped at 10,000 rows for responsiveness.")
        pcols[1].metric("Columns", f"{preview_df.shape[1]:,}")
        pcols[2].metric("Numeric columns", len(numeric_cols))
        pcols[3].metric("Missing cells in preview", f"{missing_cells:,}")
        st.markdown("#### CSV preview")
        st.dataframe(preview_df.head(25), width="stretch", hide_index=True)
        st.caption(f"Showing up to 10,000 sampled-from-start rows from `{inspect_path.name}`. Full workflow metrics come from the complete processing pipeline.")
        profile = pd.DataFrame({
            "Column": [str(col) for col in preview_df.columns],
            "Data type": [str(dtype) for dtype in preview_df.dtypes],
            "Missing %": [round(float(preview_df[col].isna().mean() * 100), 2) for col in preview_df.columns],
            "Unique values (preview)": [int(preview_df[col].nunique(dropna=True)) for col in preview_df.columns],
        }).sort_values("Missing %", ascending=False)
        left, right = st.columns([1.45, 1])
        with left:
            st.markdown("#### Column profile")
            st.dataframe(profile, width="stretch", hide_index=True, height=350)
        with right:
            st.markdown("#### Missingness overview")
            missing_plot = profile.loc[profile["Missing %"] > 0, ["Column", "Missing %"]].head(15).set_index("Column")
            if not missing_plot.empty:
                st.bar_chart(missing_plot, horizontal=True, height=350)
            else:
                st.success("No missing values were observed in this preview.")
        if cleaning_details:
            st.markdown("#### Cleaning log from the latest full workflow")
            clean_cols = st.columns(4)
            in_shape = cleaning_details.get("input_shape", [None, None])
            out_shape = cleaning_details.get("output_shape", [None, None])
            clean_cols[0].metric("Input shape", f"{in_shape[0]} × {in_shape[1]}")
            clean_cols[1].metric("Output shape", f"{out_shape[0]} × {out_shape[1]}")
            clean_cols[2].metric("Duplicate rows removed", cleaning_details.get("duplicate_rows_removed", 0))
            clean_cols[3].metric("Empty columns removed", len(cleaning_details.get("all_empty_columns_removed", [])))
            st.write("**Column-name changes:**", cleaning_details.get("column_names_trimmed", []) or "None")
            st.write("**Infinite values converted to missing:**", cleaning_details.get("infinite_values_converted_to_missing", 0))
            st.caption(cleaning_details.get("imputation_policy", ""))
            if cleaned_path is not None:
                st.download_button("Download cleaned CSV", data=cleaned_path.read_bytes(), file_name=f"cleaned_{dataset_path.name}", mime="text/csv", key="download_cleaned_csv")

with tab_visuals:
    st.markdown('<div class="section-label">Visual analytics</div>', unsafe_allow_html=True)
    st.subheader("Explore distributions, relationships, and anomalies")
    visual_choice = st.selectbox("Visualization", ["Missingness by column", "Numeric distribution", "Correlation heatmap", "Outlier overview", "Feature importance"], key="visualization_type")
    visual_source = st.radio("Analyze data from", list(inspect_options.keys()) if "inspect_options" in locals() else ["Source CSV"], horizontal=True, key="visual_source_choice")
    visual_path = (inspect_options if "inspect_options" in locals() else {"Source CSV": dataset_path})[visual_source]
    visual_df, visual_error = _read_preview(visual_path)
    if visual_error:
        st.error(visual_error)
    elif visual_df.empty:
        st.info("Upload a readable CSV to render visual analytics.")
    else:
        st.caption("Charts use a preview of up to 10,000 rows unless the analysis result below is computed by the full pipeline.")
        numeric_columns = visual_df.select_dtypes(include="number").columns.tolist()
        if visual_choice == "Missingness by column":
            missing_frame = pd.DataFrame({"Column": visual_df.columns.astype(str), "Missing %": [float(visual_df[col].isna().mean() * 100) for col in visual_df.columns]})
            missing_frame = missing_frame.sort_values("Missing %", ascending=False).head(25)
            if not missing_frame.empty and (missing_frame["Missing %"] > 0).any():
                st.bar_chart(missing_frame.set_index("Column"), horizontal=True, height=450)
            else:
                st.success("The preview has no missing values.")
        elif visual_choice == "Numeric distribution":
            if not numeric_columns:
                st.info("No numeric columns were detected in this preview.")
            else:
                selected_numeric = st.selectbox("Choose a numeric column", numeric_columns, key="distribution_numeric_column")
                series = pd.to_numeric(visual_df[selected_numeric], errors="coerce").dropna()
                if series.empty:
                    st.info("The selected column has no usable numeric values.")
                elif series.nunique() == 1:
                    st.bar_chart(series.value_counts().rename_axis("Value").to_frame("Rows"), height=350)
                else:
                    bin_count = min(24, max(5, int(len(series) ** 0.5)))
                    bins = pd.cut(series, bins=bin_count, duplicates="drop")
                    counts = bins.value_counts(sort=False)
                    counts.index = [f"{interval.left:.2f}–{interval.right:.2f}" for interval in counts.index]
                    st.bar_chart(counts.rename_axis("Value range").to_frame("Rows"), height=350)
                    st.caption(f"Preview median: {series.median():.3f} · skewness: {series.skew():.3f} · valid values: {len(series):,}")
        elif visual_choice == "Correlation heatmap":
            if len(numeric_columns) < 2:
                st.info("At least two numeric columns are needed for a correlation heatmap.")
            else:
                corr = visual_df[numeric_columns].corr(numeric_only=True)
                st.dataframe(corr.style.format("{:.2f}").background_gradient(cmap="RdYlBu_r", vmin=-1, vmax=1), width="stretch", height=500)
                st.caption("This overview uses Pearson correlation. The strategy decision panel shows where the workflow selected Pearson or Spearman for individual pairs.")
        elif visual_choice == "Outlier overview":
            outlier_rows = []
            for column in numeric_columns:
                series = pd.to_numeric(visual_df[column], errors="coerce").dropna()
                if len(series) < 4:
                    continue
                q1, q3 = series.quantile(.25), series.quantile(.75)
                iqr = q3 - q1
                count = int(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum())
                outlier_rows.append({"Column": str(column), "Potential outliers": count, "Outlier %": round(count / len(series) * 100, 2)})
            if outlier_rows:
                outlier_df = pd.DataFrame(outlier_rows).sort_values("Potential outliers", ascending=False).head(25)
                st.bar_chart(outlier_df.set_index("Column")[["Potential outliers"]], horizontal=True, height=450)
                st.dataframe(outlier_df, width="stretch", hide_index=True)
                st.caption("IQR flags are potential outliers, not proof of data errors. Calculations here use the displayed preview.")
            else:
                st.info("Not enough numeric observations to calculate IQR outliers.")
        elif visual_choice == "Feature importance":
            importance = []
            for item in results:
                details = item.get("analysis_result") or {}
                if details.get("feature_importance"):
                    importance = details["feature_importance"]
            if importance:
                imp_df = pd.DataFrame(importance)
                if "mean_accuracy_decrease" in imp_df.columns:
                    imp_df = imp_df.sort_values("mean_accuracy_decrease", ascending=True).tail(15).set_index("feature")
                    st.bar_chart(imp_df[["mean_accuracy_decrease"]], horizontal=True, height=450)
                else:
                    st.dataframe(imp_df, width="stretch", hide_index=True)
                st.caption("Permutation importance describes predictive reliance under the held-out evaluation split; it does not establish causality.")
            else:
                st.info("Feature importance appears after a full workflow with a prediction target, or after an explainability workflow has completed.")

with tab_results:
    st.markdown('<div class="section-label">Agent output</div>', unsafe_allow_html=True)
    st.subheader("Stage-specific outputs")
    if not results:
        st.info("Agent outputs appear here after a workflow completes.")
    else:
        for index, item in enumerate(results):
            task_title = _safe_text(item.get("task") or item.get("task_id") or f"Result {index + 1}", 110)
            with st.expander(f"{index + 1}. {task_title}", expanded=(index == 0)):
                st.caption(f"Agent: {_result_agent(item) or 'Not recorded'}")
                _render_agent_details(item)
                if item.get("evaluation"):
                    st.markdown("**Execution evaluation**")
                    st.json(item["evaluation"])

with tab_findings:
    st.markdown('<div class="section-label">Findings & explainability</div>', unsafe_allow_html=True)
    st.subheader("Findings, evidence, and method rationale")
    if output:
        _show_findings(output, graph_snapshot)
    else:
        st.info("Run a workflow to view finding-level evidence. Previously stored graph nodes can still be inspected in the Knowledge graph and Memory tabs.")

with tab_graph:
    st.markdown('<div class="section-label">Knowledge graph</div>', unsafe_allow_html=True)
    st.subheader("Trace how findings connect to evidence")
    if graph_snapshot is None or graph_snapshot.number_of_nodes() == 0:
        st.info("No stored graph nodes for this dataset yet. Run a workflow to create findings, evidence, and execution records.")
    else:
        graph_cols = st.columns(3)
        graph_cols[0].metric("Nodes", node_count)
        graph_cols[1].metric("Relationships", edge_count)
        graph_cols[2].metric("Node types", len({attrs.get("type", "unknown") for _, attrs in graph_snapshot.nodes(data=True)}))
        current_finding_ids = {finding_id for item in results for finding_id in item.get("finding_ids", [])}
        graph_nodes = list(graph_snapshot.nodes)
        node_options = graph_nodes[-160:]
        selected_node = st.selectbox("Focus a graph node", node_options, format_func=lambda node: _node_label(graph_snapshot, node), key="graph_focus_node")
        try:
            st.graphviz_chart(_graph_dot(graph_snapshot, current_finding_ids, focus_node=selected_node), width="stretch")
        except Exception as exc:
            st.warning("The graph renderer could not display this subgraph; node details remain available.")
            st.caption(_safe_text(exc, 280))
        left, right = st.columns([1, 1.2])
        with left:
            st.markdown("#### Selected node details")
            st.json(dict(graph_snapshot.nodes[selected_node]))
        with right:
            st.markdown("#### Direct relationships")
            relations = []
            for source, target, key, attrs in graph_snapshot.in_edges(selected_node, keys=True, data=True):
                relations.append({"Direction": "Incoming", "Related node": _node_label(graph_snapshot, source), "Relationship": attrs.get("relation", key)})
            for source, target, key, attrs in graph_snapshot.out_edges(selected_node, keys=True, data=True):
                relations.append({"Direction": "Outgoing", "Related node": _node_label(graph_snapshot, target), "Relationship": attrs.get("relation", key)})
            if relations:
                st.dataframe(pd.DataFrame(relations), width="stretch", hide_index=True)
            else:
                st.caption("No direct relationships for this node.")
        with st.expander("All graph nodes and edges"):
            node_rows = [{"ID": node_id, "Type": attrs.get("type", "unknown"), "Label": _safe_text(attrs.get("description") or attrs.get("name") or attrs.get("dataset") or attrs.get("task") or node_id, 160)} for node_id, attrs in graph_snapshot.nodes(data=True)]
            st.dataframe(pd.DataFrame(node_rows), width="stretch", hide_index=True)
            edge_rows = [{"Source": source, "Relationship": attrs.get("relation", key), "Target": target} for source, target, key, attrs in graph_snapshot.edges(keys=True, data=True)]
            st.dataframe(pd.DataFrame(edge_rows), width="stretch", hide_index=True)

with tab_memory:
    st.markdown('<div class="section-label">Self-evolution</div>', unsafe_allow_html=True)
    st.subheader("Strategy decisions and historical analytical memory")
    decisions = _strategy_decisions(output)
    graph_executions = []
    if graph_snapshot is not None:
        for node_id, attrs in graph_snapshot.nodes(data=True):
            if attrs.get("type") == "execution":
                context = attrs.get("strategy_context") or {}
                feats = context.get("features") or {}
                graph_executions.append({
                    "Execution ID": node_id,
                    "Agent": attrs.get("agent", "—"),
                    "Task": attrs.get("task", "—"),
                    "Dataset": attrs.get("dataset", "—"),
                    "Successful": attrs.get("successful", "—"),
                    "Strategy": ", ".join(attrs.get("strategies", []) or []) or "—",
                    "Rationale": "; ".join(attrs.get("strategy_reasons", []) or []) or "—",
                    "Profile version": context.get("version", "—"),
                    "Sample size": context.get("complete_sample_size", "—"),
                    "Max |skew|": round(float(feats.get("max_abs_skew", 0)), 2) if feats else "—",
                })
    strategy_bearing = [row for row in graph_executions if row["Strategy"] != "—"]
    contexts = set()
    if graph_snapshot is not None:
        for _, attrs in graph_snapshot.nodes(data=True):
            if attrs.get("type") == "execution" and attrs.get("strategy_context"):
                contexts.add(json.dumps(attrs["strategy_context"], sort_keys=True, default=str))
    mem_cols = st.columns(4)
    mem_cols[0].metric("Saved executions", len(graph_executions))
    mem_cols[1].metric("Executions with strategy", len(strategy_bearing))
    mem_cols[2].metric("Distinct stored profiles", len(contexts))
    mem_cols[3].metric("Contextual decisions this run", sum(row["Decision source"] == "Contextual memory" for row in decisions))
    st.markdown("#### Current strategy decisions")
    if decisions:
        decision_df = pd.DataFrame(decisions)
        st.dataframe(decision_df, width="stretch", hide_index=True)
        for idx, decision in enumerate(decisions, start=1):
            with st.expander(f"Decision {idx} · {decision['Column pair']} · {decision['Selected method']}"):
                st.markdown(f"**Decision source:** {decision['Decision source']}")
                st.write(decision["Rationale"])
                st.json(next((item.get("analysis_result", {}).get("strategy_context") for item in results if (item.get("analysis_result") or {}).get("columns") and " ↔ ".join(str(value) for value in item["analysis_result"].get("columns", [])) == decision["Column pair"]), {}))
    else:
        st.info("Run correlation analysis to inspect method-selection decisions. Stored history, if available, appears below.")
    if strategy_bearing:
        st.markdown("#### Strategy frequency in saved history")
        freq = {}
        for row in strategy_bearing:
            for strategy in row["Strategy"].split(", "):
                freq[strategy] = freq.get(strategy, 0) + 1
        st.bar_chart(pd.Series(freq, name="Recorded executions"), horizontal=True)
    st.markdown("#### Stored execution history")
    if graph_executions:
        st.dataframe(pd.DataFrame(graph_executions).sort_values("Execution ID", ascending=False).head(100), width="stretch", hide_index=True)
    else:
        st.info("No execution-history nodes have been recorded for this graph yet.")
    st.caption("Successful execution history is a recommendation signal, not proof of statistical superiority. Structural verification checks graph and metadata consistency, not scientific truth.")

with tab_reports:
    st.markdown('<div class="section-label">Reports & research</div>', unsafe_allow_html=True)
    st.subheader("Reports and evaluation artifacts")
    report_details = next((item.get("analysis_result") or {} for item in results if _result_agent(item) == "ReportAgent" or "report_markdown" in (item.get("analysis_result") or {})), None)
    if report_details:
        report_text = report_details.get("report_markdown") or report_details.get("markdown") or ""
        st.markdown("#### Latest workflow report")
        if report_text:
            st.markdown(report_text)
            st.download_button("Download latest report", data=report_text.encode("utf-8"), file_name="nexus_ai_analytics_report.md", mime="text/markdown", key="download_latest_report")
        else:
            st.json(report_details)
    else:
        st.info("A generated report will appear here after running the full sequential workflow or report action.")

    research_root = PROJECT_ROOT / "outputs" / "research_evaluation"
    research_reports = sorted(research_root.rglob("*.md")) if research_root.exists() else []
    research_reports = research_reports + [p for p in [PROJECT_ROOT / "research" / "CONFERENCE_RESEARCH_PLAN.md", PROJECT_ROOT / "research" / "PAPER_RESULTS_DRAFT.md"] if p.exists() and p not in research_reports]
    if research_reports:
        report_labels = [str(p.relative_to(PROJECT_ROOT)) for p in research_reports]
        chosen_report_label = st.selectbox("Open saved research report", report_labels, key="research_report_picker")
        chosen_report_path = PROJECT_ROOT / chosen_report_label
        try:
            report_body = chosen_report_path.read_text(encoding="utf-8")
            with st.expander("Preview selected research document", expanded=False):
                st.markdown(report_body)
            st.download_button("Download selected research document", data=report_body.encode("utf-8"), file_name=chosen_report_path.name, mime="text/markdown", key="download_research_doc")
        except OSError as exc:
            st.warning(f"Could not read report: {exc}")

    benchmark_candidates = {
        "Repeated-seed synthetic benchmark": research_root / "multi_seed" / "repeated_seed_summary.csv",
        "Leave-one-family-out benchmark": research_root / "leave_one_family_out" / "leave_one_family_out_method_summary.csv",
        "Real-data robustness benchmark": research_root / "real_dataset_robustness" / "real_dataset_robustness_summary.csv",
        "Initial synthetic benchmark": research_root / "method_summary.csv",
    }
    available_benchmarks = {label: p for label, p in benchmark_candidates.items() if p.exists()}
    if available_benchmarks:
        st.markdown("#### Saved benchmark comparison")
        benchmark_label = st.selectbox("Benchmark artifact", list(available_benchmarks), key="benchmark_artifact_picker")
        try:
            benchmark_df = pd.read_csv(available_benchmarks[benchmark_label])
            st.dataframe(benchmark_df, width="stretch", hide_index=True)
            if "method" in benchmark_df.columns:
                score_column = next((column for column in ["mean_dataset_macro_robustness_target_accuracy", "mean_selection_accuracy", "mean_accuracy", "selection_accuracy"] if column in benchmark_df.columns), None)
                if score_column:
                    scores = benchmark_df[["method", score_column]].copy()
                    scores[score_column] = pd.to_numeric(scores[score_column], errors="coerce") * 100
                    scores = scores.dropna().set_index("method")
                    scores.columns = ["Score (%)"]
                    st.bar_chart(scores, horizontal=True, height=350)
                    st.caption(f"Displayed metric: {score_column}; converted to percent for readability. Benchmark artifacts are historical experiment outputs, not automatically recalculated by this dashboard.")
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            st.warning(f"Could not load benchmark artifact: {exc}")

st.markdown("---")
st.caption("NEXUS AI · Context-aware analytics · Evidence-linked reasoning · Research prototype")
