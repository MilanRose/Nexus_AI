from app.knowledge_graph.nodes import (
    DatasetNode,
    FeatureNode,
    FindingNode,
    EvidenceNode,
    ModelNode,
)


def test_nodes():
    # 1. Create the five node objects
    dataset = DatasetNode(
        "dataset_1",
        "customer_churn.csv"
    )

    feature = FeatureNode(
        "feature_1",
        "MonthlyCharges"
    )

    finding = FindingNode(
        "finding_1",
        "MonthlyCharges is associated with churn",
        confidence=0.84
    )

    evidence = EvidenceNode(
        "evidence_1",
        "Pearson correlation = 0.42",
        method="Pearson correlation"
    )

    model = ModelNode(
        "model_1",
        "Random Forest",
        accuracy=0.87
    )

    # 2. Convert each node to a dictionary
    dataset_data = dataset.to_dict()
    feature_data = feature.to_dict()
    finding_data = finding.to_dict()
    evidence_data = evidence.to_dict()
    model_data = model.to_dict()

    # 3. Verify node IDs and types
    expected_nodes = [
        (dataset_data, "dataset_1", "dataset"),
        (feature_data, "feature_1", "feature"),
        (finding_data, "finding_1", "finding"),
        (evidence_data, "evidence_1", "evidence"),
        (model_data, "model_1", "model"),
    ]

    for node_data, expected_id, expected_type in expected_nodes:
        assert node_data["node_id"] == expected_id
        assert node_data["node_type"] == expected_type
        assert isinstance(node_data["attributes"], dict)

    # 4. Verify node-specific attributes
    assert dataset_data["attributes"]["name"] == "customer_churn.csv"
    assert feature_data["attributes"]["name"] == "MonthlyCharges"

    assert (
        finding_data["attributes"]["description"]
        == "MonthlyCharges is associated with churn"
    )
    assert finding_data["attributes"]["confidence"] == 0.84

    assert (
        evidence_data["attributes"]["evidence"]
        == "Pearson correlation = 0.42"
    )
    assert (
        evidence_data["attributes"]["method"]
        == "Pearson correlation"
    )

    assert model_data["attributes"]["model_name"] == "Random Forest"
    assert model_data["attributes"]["accuracy"] == 0.87

    # 5. Verify that node data remains independent
    assert dataset_data["node_id"] != feature_data["node_id"]
    assert finding_data["node_id"] != evidence_data["node_id"]

    print("NODE TEST PASSED")