import pytest

from app.knowledge_graph.builder import KnowledgeGraphBuilder
from app.knowledge_graph.evolution import SelfEvolutionEngine
from app.knowledge_graph.graph import KnowledgeGraph
from app.schemas import AgentResult, Evidence, ModelResult


def make_result(finding="MonthlyCharges is associated with churn"):
    return AgentResult(
        agent="EDAAgent",
        dataset="customer_churn.csv",
        task="correlation_analysis",
        findings=[finding],
        features=["MonthlyCharges"],
        evidence=[
            Evidence(
                description="Pearson correlation = 0.42",
                method="Pearson correlation",
                value=0.42,
                finding_indices=[0],
            )
        ],
        model=ModelResult(
            name="Random Forest",
            metrics={"accuracy": 0.87},
        ),
        confidence=0.84,
    )


def make_evaluation():
    return {
        "successful": True,
        "confidence": 0.84,
        "model_performance": 0.87,
        "verification_passed": True,
        "checks": {"all_findings_verified": True},
    }


def test_builder_retry_reuses_existing_nodes():
    kg = KnowledgeGraph()
    builder = KnowledgeGraphBuilder(kg)
    result = make_result()

    first_ids = builder.add_agent_result(result, run_id="run-001")
    node_count = kg.graph.number_of_nodes()
    edge_count = kg.graph.number_of_edges()

    retry_ids = builder.add_agent_result(result, run_id="run-001")

    assert retry_ids == first_ids
    assert kg.graph.number_of_nodes() == node_count
    assert kg.graph.number_of_edges() == edge_count


def test_builder_rejects_conflicting_result_for_same_run():
    kg = KnowledgeGraph()
    builder = KnowledgeGraphBuilder(kg)

    builder.add_agent_result(make_result(), run_id="run-002")
    original_nodes = dict(kg.graph.nodes(data=True))
    original_edges = list(kg.graph.edges(keys=True, data=True))

    with pytest.raises(ValueError, match="different analysis results"):
        builder.add_agent_result(
            make_result("A different finding"),
            run_id="run-002",
        )

    assert dict(kg.graph.nodes(data=True)) == original_nodes
    assert list(kg.graph.edges(keys=True, data=True)) == original_edges


def test_evolution_retry_reuses_execution_history():
    kg = KnowledgeGraph()
    builder = KnowledgeGraphBuilder(kg)
    evolution = SelfEvolutionEngine(kg)

    result = make_result()
    evaluation = make_evaluation()

    finding_ids = builder.add_agent_result(
        result, run_id="run-003:task-1"
    )

    first_id = evolution.update_knowledge(
        result, evaluation, finding_ids, run_id="run-003:task-1"
    )
    node_count = kg.graph.number_of_nodes()
    edge_count = kg.graph.number_of_edges()

    retry_id = evolution.update_knowledge(
        result, evaluation, finding_ids, run_id="run-003:task-1"
    )

    assert retry_id == first_id
    assert kg.graph.number_of_nodes() == node_count
    assert kg.graph.number_of_edges() == edge_count


def test_new_run_creates_separate_execution_history():
    kg = KnowledgeGraph()
    builder = KnowledgeGraphBuilder(kg)
    evolution = SelfEvolutionEngine(kg)

    result = make_result()
    evaluation = make_evaluation()

    first_findings = builder.add_agent_result(
        result, run_id="run-004:task-1"
    )
    first_execution = evolution.update_knowledge(
        result, evaluation, first_findings, run_id="run-004:task-1"
    )

    second_findings = builder.add_agent_result(
        result, run_id="run-005:task-1"
    )
    second_execution = evolution.update_knowledge(
        result, evaluation, second_findings, run_id="run-005:task-1"
    )

    assert first_execution != second_execution
    assert first_findings != second_findings

    executions = [
        data
        for _, data in kg.graph.nodes(data=True)
        if data.get("type") == "execution"
    ]
    assert len(executions) == 2