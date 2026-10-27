from app.agents.data_agent import DataAgent


def test_data_agent_profiles_dataset_and_generates_evidence(tmp_path):
    # Create a small CSV with missing values and one duplicate row.
    dataset_path = tmp_path / "sample.csv"
    dataset_path.write_text(
        "id,age,city\n"
        "1,25,Alpha\n"
        "2,,Beta\n"
        "2,,Beta\n"
        "3,40,Alpha\n",
        encoding="utf-8",
    )

    agent = DataAgent()

    result, profile = agent.run(str(dataset_path))

    # Check the shared result structure.
    assert result.agent == "DataAgent"
    assert result.dataset == "sample.csv"
    assert result.task == "data_quality_profile"
    assert result.model is None
    assert result.confidence > 0

    # Check the dataset profile.
    assert profile["n_rows"] == 4
    assert profile["n_cols"] == 3
    assert profile["duplicate_rows"] == 1
    assert profile["missing_values"]["age"] == 2

    # Check the generated findings.
    assert any(
        "4 rows and 3 columns" in finding
        for finding in result.findings
    )

    assert any(
        "age has 2 missing values" in finding
        for finding in result.findings
    )

    assert any(
        "1 duplicate rows" in finding
        for finding in result.findings
    )

    # Every finding should have corresponding evidence.
    assert result.findings
    assert result.evidence

    for finding_index in range(len(result.findings)):
        matching_evidence = [
            item
            for item in result.evidence
            if finding_index in item.finding_indices
        ]
        assert matching_evidence, (
            f"Finding {finding_index} has no linked evidence."
        )

    print("DataAgent profiling test passed.")


def test_data_agent_rejects_missing_dataset(tmp_path):
    missing_path = tmp_path / "does_not_exist.csv"

    agent = DataAgent()

    try:
        agent.run(str(missing_path))
    except (FileNotFoundError, OSError):
        pass
    else:
        raise AssertionError(
            "DataAgent should reject a missing dataset."
        )