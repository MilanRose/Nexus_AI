from app.orchestrator.orchestrator import Orchestrator


def test_data_agent_request_integrates_with_knowledge_graph(tmp_path):
    dataset_path = tmp_path / "sample.csv"
    graph_path = tmp_path / "knowledge_graph.json"

    # Create a CSV with one missing value and one duplicate row.
    rows = ["id,age,city"]

    for i in range(100):
        age = "" if i == 10 else str(20 + i)
        rows.append(f"{i},{age},Alpha")

    rows.append(rows[-1])

    dataset_path.write_text(
        "\n".join(rows) + "\n",
        encoding="utf-8",
    )

    orchestrator = Orchestrator(
        dataset_path=dataset_path,
        graph_path=graph_path,
    )

    output = orchestrator.process_request(
        "Inspect this dataset for missing values and duplicate rows"
    )

    assert output["selected_agent"] == "DataAgent"
    assert output["status"] == "completed"
    assert output["task_count"] == 1
    assert len(output["results"]) == 1

    result = output["results"][0]

    assert result["finding_ids"]
    assert result["findings"]
    assert result["analysis_result"]["n_rows"] == 101
    assert result["analysis_result"]["duplicate_rows"] == 1
    assert result["analysis_result"]["missing_values"]["age"] == 1

    # Verify structure without claiming scientific support.
    assert all(
        check["structural_verified"] is True
        for check in result["verification"]
    )
    assert all(
        check["verified"] is False
        for check in result["verification"]
    )

    assert result["evaluation"]["successful"] is True
    assert graph_path.exists()

    # Confirm execution history was recorded.
    execution_nodes = [
        data
        for _, data in orchestrator.knowledge_graph.graph.nodes(
            data=True
        )
        if data.get("type") == "execution"
    ]

    assert execution_nodes