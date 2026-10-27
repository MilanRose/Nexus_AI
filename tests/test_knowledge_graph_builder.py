import pytest

from app.knowledge_graph.builder import KnowledgeGraphBuilder
from app.knowledge_graph.graph import KnowledgeGraph
from app.schemas import AgentResult, Evidence, ModelResult


@pytest.fixture
def result():
    return AgentResult(
        agent="EDAAgent",
        dataset="sample.csv",
        task="correlation_analysis",
        findings=["MonthlyCharges is associated with churn"],
        features=["MonthlyCharges"],
        evidence=[
            Evidence(
                description="Pearson correlation = 0.42",
                method="pearson",
                value=0.42,
                finding_indices=[0],
            )
        ],
        model=ModelResult(
            name="RandomForest",
            metrics={"accuracy": 0.87},
        ),
        confidence=0.84,
    )


@pytest.mark.parametrize(
    "colliding_id",
    ["model_1_1", "evidence_1_1"],
)
def test_builder_avoids_existing_model_and_evidence_ids(
    result, colliding_id
):
    kg = KnowledgeGraph()

    # Simulate an existing node whose ID must not be overwritten.
    kg.add_node(
        colliding_id,
        "existing",
        marker="preserve this node",
    )

    builder = KnowledgeGraphBuilder(kg)
    finding_ids = builder.add_agent_result(result)

    # The builder should skip result number 1.
    assert finding_ids == ["finding_2_1"]
    assert "model_2_1" in kg.graph
    assert "evidence_2_1" in kg.graph

    # The previously existing node must remain unchanged.
    existing_node = kg.get_node(colliding_id)
    assert existing_node["marker"] == "preserve this node"
    assert existing_node["type"] == "existing"


def test_builder_uses_first_available_result_number(result):
    kg = KnowledgeGraph()
    builder = KnowledgeGraphBuilder(kg)

    finding_ids = builder.add_agent_result(result)

    assert finding_ids == ["finding_1_1"]
    assert "model_1_1" in kg.graph
    assert "evidence_1_1" in kg.graph