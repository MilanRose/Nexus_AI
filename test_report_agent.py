
from app.agents.report_agent import ReportAgent
from app.knowledge_graph.graph import KnowledgeGraph


def test_report_agent_generates_evidence_aware_report():
    kg = KnowledgeGraph()

    # Dataset and finding
    kg.add_node(
        "dataset_titanic.csv",
        "dataset",
        name="titanic.csv",
    )

    kg.add_node(
        "finding_1_1",
        "finding",
        description=(
            "The stored analysis reports a correlation of 0.42."
        ),
        agent="EDAAgent",
        task="investigate_correlation",
        confidence=0.81,
    )

    kg.add_relationship(
        "dataset_titanic.csv",
        "contains_finding",
        "finding_1_1",
    )

    # Evidence connected to the finding
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

    # Model metadata connected to the finding
    kg.add_node(
        "model_1_1",
        "model",
        model_name="RandomForestClassifier",
        accuracy=0.84,
        metrics={
            "accuracy": 0.84,
            "f1_macro": 0.82,
        },
    )

    kg.add_relationship(
        "finding_1_1",
        "generated_by",
        "model_1_1",
    )

    # Execution history connected to the finding
    kg.add_node(
        "execution_1",
        "execution",
        agent="EDAAgent",
        task="investigate_correlation",
        dataset="titanic.csv",
        successful=True,
        confidence=0.81,
        model_performance=0.84,
        verification_passed=True,
    )

    kg.add_relationship(
        "execution_1",
        "produced",
        "finding_1_1",
    )

    agent = ReportAgent()

    report = agent.run(
        kg,
        dataset_name="titanic.csv",
    )

    assert report["agent"] == "ReportAgent"
    assert report["status"] == "completed"
    assert report["finding_count"] == 1

    finding = report["findings"][0]

    assert finding["finding_id"] == "finding_1_1"
    assert len(finding["evidence"]) == 1
    assert finding["evidence"][0]["value"] == 0.42

    assert len(finding["models"]) == 1
    assert (
        finding["models"][0]["model_name"]
        == "RandomForestClassifier"
    )

    assert report["execution_history"]["count"] == 1
    assert report["execution_history"]["successful_count"] == 1

    assert "0.42" in report["report_markdown"]
    assert "Pearson correlation" in report["report_markdown"]
    assert "RandomForestClassifier" in report["report_markdown"]

    # The report must clearly distinguish stored evidence from
    # scientific verification.
    assert any(
        "scientific verification" in limitation.lower()
        for limitation in report["limitations"]
    )


def test_report_agent_handles_dataset_without_findings():
    kg = KnowledgeGraph()

    kg.add_node(
        "dataset_empty.csv",
        "dataset",
        name="empty.csv",
    )

    agent = ReportAgent()

    report = agent.run(
        kg,
        dataset_name="empty.csv",
    )

    assert report["agent"] == "ReportAgent"
    assert report["status"] == "no_findings"
    assert report["finding_count"] == 0
    assert report["findings"] == []
    assert "No findings" in report["report_markdown"]
