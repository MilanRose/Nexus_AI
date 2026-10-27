
import pandas as pd

from app.orchestrator.orchestrator import Orchestrator


def test_ml_agent_runs_through_orchestrator(tmp_path):
    dataset_path = tmp_path / "classification.csv"
    graph_path = tmp_path / "knowledge_graph.json"

    df = pd.DataFrame(
        {
            "PassengerId": range(1, 121),
            "signal": [i % 2 for i in range(120)],
            "amount": [
                float(i % 2) * 10.0 + (i % 5)
                for i in range(120)
            ],
            "category": [
                "high" if i % 2 else "low"
                for i in range(120)
            ],
            "Survived": [i % 2 for i in range(120)],
        }
    )

    df.to_csv(dataset_path, index=False)

    orchestrator = Orchestrator(
        dataset_path=dataset_path,
        graph_path=graph_path,
    )

    output = orchestrator.process_request(
        "Predict survival using a machine learning model",
        target_column="Survived",
    )

    assert output["selected_agent"] == "MLAgent"
    assert output["status"] == "completed"
    assert len(output["results"]) == 1

    result = output["results"][0]
    details = result["analysis_result"]

    assert details["target_column"] == "Survived"
    assert details["model_name"] == "RandomForestClassifier"
    assert 0.0 <= details["metrics"]["accuracy"] <= 1.0

    assert result["finding_ids"]
    assert result["findings"]
    assert result["verification"]
    assert result["evaluation"] is not None

    assert graph_path.exists()

    # Structural checks must pass, but scientific verification
    # must not be falsely claimed.
    assert all(
        item["structural_verified"] is True
        for item in result["verification"]
    )

    assert all(
        item["verified"] is False
        for item in result["verification"]
    )

    # Ensure execution history was added to the knowledge graph.
    execution_nodes = [
        data
        for _, data in orchestrator.knowledge_graph.graph.nodes(
            data=True
        )
        if data.get("type") == "execution"
    ]

    assert execution_nodes


def test_ml_agent_requests_target_column(tmp_path):
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

    output = orchestrator.process_request(
        "Predict the label using a classification model"
    )

    assert output["selected_agent"] == "MLAgent"
    assert output["status"] == "input_required"
    assert "target column" in output["message"].lower()
    assert output["results"] == []
