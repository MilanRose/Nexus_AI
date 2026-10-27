
import pandas as pd

from app.orchestrator.orchestrator import Orchestrator


def test_report_agent_runs_through_orchestrator(tmp_path):
    dataset_path = tmp_path / "sample.csv"
    graph_path = tmp_path / "knowledge_graph.json"

    pd.DataFrame(
        {
            "feature": range(20),
            "label": [i % 2 for i in range(20)],
        }
    ).to_csv(dataset_path, index=False)

    orchestrator = Orchestrator(
        dataset_path=dataset_path,
        graph_path=graph_path,
    )

    kg = orchestrator.knowledge_graph

    kg.add_node(
        "dataset_sample.csv",
        "dataset",
        name="sample.csv",
    )

    kg.add_node(
        "finding_1_1",
        "finding",
        description="The stored analysis reports a correlation of 0.42.",
        agent="EDAAgent",
        task="investigate_correlation",
        confidence=0.81,
    )

    kg.add_relationship(
        "dataset_sample.csv",
        "contains_finding",
        "finding_1_1",
    )

    kg.add_node(
        "evidence_1_1",
        "evidence",
        description="Recorded Pearson correlation",
        method="Pearson correlation",
        value=0.42,
    )

    kg.add_relationship(
        "finding_1_1",
        "supported_by",
        "evidence_1_1",
    )

    output = orchestrator.process_request(
        "Generate a report explaining the dataset findings"
    )

    assert output["selected_agent"] == "ReportAgent"
    assert output["status"] == "completed"
    assert len(output["results"]) == 1

    report = output["results"][0]["analysis_result"]

    assert report["agent"] == "ReportAgent"
    assert report["finding_count"] == 1
    assert report["findings"][0]["finding_id"] == "finding_1_1"

    assert "0.42" in report["report_markdown"]
    assert "Pearson correlation" in report["report_markdown"]

    # Reporting must not claim scientific verification.
    assert any(
        "scientific verification" in item.lower()
        for item in report["limitations"]
    )

    assert output["errors"] == []
