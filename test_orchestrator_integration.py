
import pandas as pd
import pytest 

from app.orchestrator.orchestrator import Orchestrator


def create_test_dataset(tmp_path):
    """Create a small dataset for orchestrator integration tests."""
    dataset_path = tmp_path / "test_dataset.csv"

    df = pd.DataFrame(
        {
            "PassengerId": range(1, 41),
            "feature": range(40),
            "amount": [
                float(i * 2) for i in range(40)
            ],
            "label": [
                i % 2 for i in range(40)
            ],
        }
    )

    df.to_csv(dataset_path, index=False)
    return dataset_path


def create_orchestrator(tmp_path):
    """Create an orchestrator using temporary test files."""
    dataset_path = create_test_dataset(tmp_path)
    graph_path = tmp_path / "knowledge_graph.json"

    return Orchestrator(
        dataset_path=dataset_path,
        graph_path=graph_path,
    )


def assert_response_fields(output):
    """Check that the response has the expected structure."""
    expected_fields = {
        "user_request",
        "selected_agent",
        "status",
        "message",
        "dataset",
        "task_count",
        "insights",
        "results",
        "errors",
        "run_id",
    }

    assert expected_fields.issubset(output.keys())


def test_data_agent_runs_through_orchestrator(tmp_path):
    orchestrator = create_orchestrator(tmp_path)

    output = orchestrator.process_request(
        "Inspect the dataset for missing values"
    )

    assert output["selected_agent"] == "DataAgent"
    assert output["status"] in {
        "completed",
        "completed_with_warnings",
    }

    assert_response_fields(output)

    assert len(output["results"]) == 1
    assert output["results"][0]["findings"]
    assert output["results"][0]["finding_ids"]


def test_ml_agent_requests_target_column(tmp_path):
    orchestrator = create_orchestrator(tmp_path)

    output = orchestrator.process_request(
        "Predict the label using a classification model"
    )

    assert output["selected_agent"] == "MLAgent"
    assert output["status"] == "input_required"
    assert "target column" in output["message"].lower()

    assert output["results"] == []
    assert_response_fields(output)


def test_report_agent_has_consistent_response_fields(tmp_path):
    orchestrator = create_orchestrator(tmp_path)

    output = orchestrator.process_request(
        "Generate a report explaining the dataset findings"
    )

    assert output["selected_agent"] == "ReportAgent"
    assert output["status"] == "no_findings"
    assert_response_fields(output)

    # ReportAgent is implemented, so it returns a report even when
    # the graph contains no findings.
    assert len(output["results"]) == 1
    assert output["errors"] == []
    assert output["run_id"]

    report = output["results"][0]["analysis_result"]

    assert report["agent"] == "ReportAgent"
    assert report["status"] == "no_findings"
    assert report["finding_count"] == 0
    assert report["findings"] == []
    assert "No findings" in report["report_markdown"]


def test_empty_request_is_rejected(tmp_path):
    orchestrator = create_orchestrator(tmp_path)

    with pytest.raises(ValueError):
        orchestrator.process_request("")