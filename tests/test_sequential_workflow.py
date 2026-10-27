from __future__ import annotations

import numpy as np
import pandas as pd

from app.agents.cleaning_agent import DataCleaningAgent
from app.orchestrator.orchestrator import Orchestrator


def make_classification_csv(path, *, rows=100):
    rng = np.random.default_rng(17)
    x = rng.normal(size=rows)
    frame = pd.DataFrame(
        {
            " x ": x,
            "linear_y": 2.5 * x + rng.normal(0, 0.08, size=rows),
            "nonlinear_y": x**2,
            "kind": np.where(x > 0, "positive", "negative"),
            "target": (x > 0).astype(int),
            "partial_missing": [np.nan if i % 5 == 0 else float(i % 7) for i in range(rows)],
            "all_empty": [np.nan] * rows,
        }
    )
    frame.loc[rows - 1] = frame.loc[rows - 2]
    frame.to_csv(path, index=False)
    return frame


def test_cleaning_agent_creates_separate_traceable_artifact(tmp_path):
    source = tmp_path / "source.csv"
    original = make_classification_csv(source, rows=40)
    source_before = source.read_bytes()

    result, details, cleaned_path = DataCleaningAgent().run(
        str(source), output_root=tmp_path / "cleaned"
    )

    cleaned = pd.read_csv(cleaned_path)
    assert source.read_bytes() == source_before
    assert cleaned_path.exists()
    assert cleaned_path.name == source.name
    assert "x" in cleaned.columns
    assert "all_empty" not in cleaned.columns
    assert cleaned["partial_missing"].isna().any()
    assert len(cleaned) == len(original.drop_duplicates())
    assert details["duplicate_rows_removed"] == 1
    assert details["all_empty_columns_removed"] == ["all_empty"]
    assert details["missing_values_preserved_for_eda"] is True
    assert result.agent == "DataCleaningAgent"
    assert len(result.findings) == len(result.evidence) >= 4
    assert cleaned_path.with_suffix(".manifest.json").exists()


def test_full_sequential_workflow_runs_clean_eda_ml_exact_model_xai_and_report(tmp_path):
    source = tmp_path / "sample.csv"
    make_classification_csv(source, rows=100)
    graph_path = tmp_path / "graph.json"

    orchestrator = Orchestrator(
        dataset_path=str(source),
        graph_path=graph_path,
    )
    output = orchestrator.process_request(
        "run full sequential analytics workflow",
        target_column="target",
    )

    assert output["status"] == "completed", output["errors"]
    assert "cleaning → EDA → ML → explainability" in output["message"]
    task_names = [item["task"] for item in output["results"]]
    assert any("Data cleaning" in task for task in task_names)
    assert any("Stage 2 — EDA" in task for task in task_names)
    assert any("ML training and evaluation" in task for task in task_names)
    assert any("Explainability of fitted model" in task for task in task_names)
    assert any("Final report" in task for task in task_names)

    ml_stage = next(
        item for item in output["results"]
        if item.get("agent") == "MLAgent"
    )
    xai_stage = next(
        item for item in output["results"]
        if item.get("agent") == "ExplainabilityAgent"
    )
    ml_details = ml_stage["analysis_result"]
    xai_details = xai_stage["analysis_result"]
    assert xai_details["model_source"] == "exact_fitted_model_from_MLAgent"
    assert xai_details["test_samples"] == ml_details["test_samples"]
    assert xai_details["baseline_accuracy"] == ml_details["metrics"]["accuracy"]
    assert xai_stage["finding_ids"]
    assert graph_path.exists()
    assert output["task_count"] >= 4

    # Reloading the persisted graph must preserve records from each stage.
    reloaded = Orchestrator(dataset_path=str(source), graph_path=graph_path)
    node_agents = {
        attrs.get("agent")
        for _, attrs in reloaded.knowledge_graph.graph.nodes(data=True)
        if attrs.get("type") == "execution"
    }
    assert {"DataCleaningAgent", "EDAAgent", "MLAgent", "ExplainabilityAgent"}.issubset(node_agents)


def test_full_sequential_workflow_can_skip_predictive_stages(tmp_path):
    source = tmp_path / "unsupervised.csv"
    rng = np.random.default_rng(9)
    x = rng.normal(size=70)
    pd.DataFrame(
        {
            "x": x,
            "y": 1.8 * x + rng.normal(0, 0.05, size=len(x)),
            "group": np.where(x > 0, "A", "B"),
        }
    ).to_csv(source, index=False)

    output = Orchestrator(
        dataset_path=str(source),
        graph_path=tmp_path / "graph.json",
    ).process_request("run full sequential analytics workflow")

    assert output["status"] in {"completed", "completed_with_warnings"}
    assert any("Data cleaning" in item["task"] for item in output["results"])
    assert any("Stage 2 — EDA" in item["task"] for item in output["results"])
    assert not any(item.get("agent") == "MLAgent" for item in output["results"])
    assert not any(item.get("agent") == "ExplainabilityAgent" for item in output["results"])
    assert any(item.get("agent") == "ReportAgent" for item in output["results"])
