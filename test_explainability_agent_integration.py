
import pandas as pd

from app.orchestrator.orchestrator import Orchestrator


def create_dataset(path, rows=120):
    """Create a reproducible classification dataset."""

    df = pd.DataFrame(
        {
            "PassengerId": range(1, rows + 1),
            "signal": [i % 2 for i in range(rows)],
            "noise": [
                float((i * 17) % 23) / 23.0
                for i in range(rows)
            ],
            "category": [
                "even" if i % 2 == 0 else "odd"
                for i in range(rows)
            ],
            "Survived": [i % 2 for i in range(rows)],
        }
    )

    df.to_csv(path, index=False)
    return df


def test_explainability_agent_runs_through_orchestrator(
    tmp_path,
):
    dataset_path = tmp_path / "classification.csv"
    graph_path = tmp_path / "knowledge_graph.json"

    create_dataset(dataset_path)

    orchestrator = Orchestrator(
        dataset_path=dataset_path,
        graph_path=graph_path,
    )

    output = orchestrator.process_request(
        "Explain feature importance for the survival model",
        target_column="Survived",
    )

    assert output["selected_agent"] == "ExplainabilityAgent"
    assert output["status"] == "completed"
    assert len(output["results"]) == 1

    result = output["results"][0]
    details = result["analysis_result"]

    assert details["agent"] == "ExplainabilityAgent"
    assert details["target_column"] == "Survived"

    assert (
        details["explanation_method"]
        == "permutation_importance"
    )

    assert (
        details["model_source"]
        == "refitted_reference_model"
    )

    assert 0.0 <= details["baseline_accuracy"] <= 1.0
    assert 0.0 <= details["f1_macro"] <= 1.0

    assert details["feature_importance"]
    assert result["finding_ids"]
    assert result["findings"]
    assert result["verification"]
    assert result["evaluation"] is not None

    assert graph_path.exists()

    # Findings should be stored in the knowledge graph.
    for finding_id in result["finding_ids"]:
        node = orchestrator.knowledge_graph.get_node(
            finding_id
        )

        assert node["type"] == "finding"
        assert node["agent"] == "ExplainabilityAgent"

    # Every returned finding should have linked evidence.
    for finding_id in result["finding_ids"]:
        evidence_targets = [
            target
            for _, target, data in (
                orchestrator.knowledge_graph.graph.out_edges(
                    finding_id,
                    data=True,
                )
            )
            if data.get("relation") == "supported_by"
        ]

        assert evidence_targets


def test_explainability_agent_requires_target_column(
    tmp_path,
):
    dataset_path = tmp_path / "classification.csv"
    graph_path = tmp_path / "knowledge_graph.json"

    create_dataset(dataset_path, rows=30)

    orchestrator = Orchestrator(
        dataset_path=dataset_path,
        graph_path=graph_path,
    )

    output = orchestrator.process_request(
        "Explain feature importance for this model"
    )

    assert output["selected_agent"] == "ExplainabilityAgent"
    assert output["status"] == "input_required"
    assert "target column" in output["message"].lower()
    assert output["results"] == []


def test_router_keeps_report_requests_separate():
    from app.orchestrator.router import AdaptiveRouter

    router = AdaptiveRouter()

    assert (
        router.route(
            "Generate a report explaining the dataset findings"
        )
        == "ReportAgent"
    )

    assert (
        router.route(
            "Explain feature importance for a classification model"
        )
        == "ExplainabilityAgent"
    )

    assert (
        router.route(
            "Predict survival using a classification model"
        )
        == "MLAgent"
    )
