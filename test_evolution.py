from app.knowledge_graph.graph import KnowledgeGraph
from app.knowledge_graph.builder import KnowledgeGraphBuilder
from app.knowledge_graph.evolution import SelfEvolutionEngine
from app.schemas import AgentResult, Evidence, ModelResult


def create_result(
    agent,
    task,
    finding,
    evidence_description,
    confidence,
    accuracy,
    dataset="customer_churn.csv"
):
    return AgentResult(
        agent=agent,
        dataset=dataset,
        task=task,
        findings=[finding],
        evidence=[
            Evidence(
                description=evidence_description,
                method="Test evidence",
                value=0.42,
                finding_indices=[0]
            )
        ],
        model=ModelResult(
            name="Random Forest",
            metrics={"accuracy": accuracy}
        ),
        confidence=confidence
    )


def store_execution(kg, builder, evolution, result, verified=True):
    finding_ids = builder.add_agent_result(result)

    verification = {
        "verified": verified,
        "checks": {
            "finding_exists": True,
            "has_confidence": True,
            "has_evidence": verified,
            "has_model": True,
            "valid_confidence": True,
            "valid_model_accuracy": True
        }
    }

    evaluation = evolution.evaluate_execution(result, verification)
    execution_id = evolution.update_knowledge(
        result, evaluation, finding_ids
    )

    return execution_id, finding_ids


def test_evolution():
    kg = KnowledgeGraph()
    builder = KnowledgeGraphBuilder(kg)
    evolution = SelfEvolutionEngine(kg)

    # 1. Store a successful EDA execution
    result = create_result(
        agent="EDAAgent",
        task="correlation_analysis",
        finding="MonthlyCharges is associated with churn",
        evidence_description="Pearson correlation = 0.42",
        confidence=0.84,
        accuracy=0.87
    )

    execution_id, finding_ids = store_execution(
        kg, builder, evolution, result
    )

    assert kg.graph.has_edge(execution_id, finding_ids[0])
    assert any(
        target == finding_ids[0] and data.get("relation") == "produced"
        for _, target, data in kg.graph.out_edges(execution_id, data=True)
    )

    # 2. Store a failed ML execution for the same task and dataset
    failed_result = create_result(
        agent="MLAgent",
        task="correlation_analysis",
        finding="Test finding from MLAgent",
        evidence_description="Weak evidence",
        confidence=0.4,
        accuracy=0.4
    )

    failed_execution_id, failed_finding_ids = store_execution(
        kg, builder, evolution, failed_result, verified=False
    )

    assert kg.graph.has_edge(failed_execution_id, failed_finding_ids[0])

    # 3. Verify exact provenance
    assert not kg.graph.has_edge(execution_id, failed_finding_ids[0])
    assert not kg.graph.has_edge(failed_execution_id, finding_ids[0])

    # 4. Verify recommendation for the existing task and dataset
    recommendation = evolution.recommend_workflow(
        "correlation_analysis",
        dataset="customer_churn.csv"
    )

    assert recommendation["recommended_agent"] == "EDAAgent"
    assert recommendation["confidence"] == 0.84
    assert recommendation["model_performance"] == 0.87

    # 5. Regression test: another dataset must not affect the recommendation
    other_result = create_result(
        agent="OtherDatasetAgent",
        task="correlation_analysis",
        finding="Finding from another dataset",
        evidence_description="Other dataset evidence",
        confidence=0.99,
        accuracy=0.99,
        dataset="different_dataset.csv"
    )

    store_execution(kg, builder, evolution, other_result)

    isolated_recommendation = evolution.recommend_workflow(
        "correlation_analysis",
        dataset="customer_churn.csv"
    )

    assert isolated_recommendation["recommended_agent"] == "EDAAgent"
    assert isolated_recommendation["confidence"] == 0.84

    assert evolution.get_recommended_agent(
        "correlation_analysis",
        dataset="customer_churn.csv"
    ) == "EDAAgent"

    assert evolution.find_similar_executions(
        "correlation_analysis",
        dataset="customer_churn.csv"
    )
    assert all(
        execution["dataset"] == "customer_churn.csv"
        for execution in evolution.find_similar_executions(
            "correlation_analysis",
            dataset="customer_churn.csv"
        )
    )

    # 6. Without a dataset filter, legacy calls remain supported

    # 6. Without a dataset filter, ambiguous history must not be mixed.
    legacy_recommendation = evolution.recommend_workflow(
        "correlation_analysis"
    )

    assert legacy_recommendation["recommended_agent"] is None
    assert "Multiple datasets" in legacy_recommendation["reason"]


    print("EVOLUTION AND DATASET-ISOLATION TEST PASSED")

def test_recommendation_without_execution_history():
    kg = KnowledgeGraph()
    evolution = SelfEvolutionEngine(kg)

    recommendation = evolution.recommend_workflow(
        "correlation_analysis"
    )

    assert recommendation["recommended_agent"] is None
    assert recommendation["dataset"] is None
    assert "No previous execution" in recommendation["reason"]


def test_recommendation_prefers_higher_model_performance():
    kg = KnowledgeGraph()
    builder = KnowledgeGraphBuilder(kg)
    evolution = SelfEvolutionEngine(kg)

    first_result = create_result(
        agent="EDAAgent",
        task="correlation_analysis",
        finding="Finding from EDA",
        evidence_description="EDA evidence",
        confidence=0.95,
        accuracy=0.80
    )

    store_execution(
        kg, builder, evolution, first_result
    )

    second_result = create_result(
        agent="MLAgent",
        task="correlation_analysis",
        finding="Finding from ML",
        evidence_description="ML evidence",
        confidence=0.80,
        accuracy=0.92
    )

    store_execution(
        kg, builder, evolution, second_result
    )

    recommendation = evolution.recommend_workflow(
        "correlation_analysis",
        dataset="customer_churn.csv"
    )

    assert recommendation["recommended_agent"] == "MLAgent"
    assert recommendation["model_performance"] == 0.92

def test_recovered_agent_can_be_recommended_after_later_success():
    kg = KnowledgeGraph()
    builder = KnowledgeGraphBuilder(kg)
    evolution = SelfEvolutionEngine(kg)

    # First, EDAAgent fails.
    failed_result = create_result(
        agent="EDAAgent",
        task="correlation_analysis",
        finding="Initial unsuccessful analysis",
        evidence_description="Insufficient evidence",
        confidence=0.4,
        accuracy=0.4,
    )

    store_execution(
        kg, builder, evolution, failed_result, verified=False
    )

    # The same agent later succeeds at the same task and dataset.
    recovered_result = create_result(
        agent="EDAAgent",
        task="correlation_analysis",
        finding="Recovered correlation analysis",
        evidence_description="Pearson correlation = 0.42",
        confidence=0.84,
        accuracy=0.87,
    )

    store_execution(
        kg, builder, evolution, recovered_result, verified=True
    )

    recommendation = evolution.recommend_workflow(
        "correlation_analysis",
        dataset="customer_churn.csv",
    )

    assert recommendation["recommended_agent"] == "EDAAgent"
    assert recommendation["confidence"] == 0.84
    assert recommendation["model_performance"] == 0.87
    assert recommendation["failed_agents"] == []


def test_recommendation_does_not_compare_accuracy_with_confidence():
    kg = KnowledgeGraph()
    builder = KnowledgeGraphBuilder(kg)
    evolution = SelfEvolutionEngine(kg)

    # This agent has measured model performance.
    accuracy_result = create_result(
        agent="AccuracyAgent",
        task="correlation_analysis",
        finding="Analysis with measured model performance",
        evidence_description="Model evaluation evidence",
        confidence=0.75,
        accuracy=0.75,
    )

    store_execution(
        kg, builder, evolution, accuracy_result
    )

    # This agent has high confidence but no measured model accuracy.
    confidence_result = AgentResult(
        agent="ConfidenceOnlyAgent",
        dataset="customer_churn.csv",
        task="correlation_analysis",
        findings=["Analysis without a model"],
        evidence=[
            Evidence(
                description="Analysis evidence",
                method="EDA",
                value=0.9,
                finding_indices=[0],
            )
        ],
        model=None,
        confidence=0.99,
    )

    store_execution(
        kg, builder, evolution, confidence_result
    )

    recommendation = evolution.recommend_workflow(
        "correlation_analysis",
        dataset="customer_churn.csv",
    )

    assert recommendation["recommended_agent"] == "AccuracyAgent"
    assert recommendation["model_performance"] == 0.75
    assert "model performance" in recommendation["reason"]
    
def test_execution_history_records_analysis_strategies():
    kg = KnowledgeGraph()
    builder = KnowledgeGraphBuilder(kg)
    evolution = SelfEvolutionEngine(kg)

    result = AgentResult(
        agent="EDAAgent",
        dataset="customer_churn.csv",
        task="correlation_analysis",
        findings=["MonthlyCharges is associated with churn"],
        evidence=[
            Evidence(
                description="Pearson correlation = 0.42",
                method="Pearson correlation",
                value=0.42,
                finding_indices=[0],
            ),
            Evidence(
                description="Spearman correlation = 0.39",
                method="Spearman correlation",
                value=0.39,
                finding_indices=[0],
            ),
        ],
        model=None,
        confidence=0.84,
    )

    finding_ids = builder.add_agent_result(result)
    evaluation = evolution.evaluate_execution(
        result,
        {
            "structural_verified": True,
            "checks": {"has_evidence": True},
        },
    )

    execution_id = evolution.update_knowledge(
        result,
        evaluation,
        finding_ids,
    )

    history = evolution.find_similar_executions(
        "correlation_analysis",
        dataset="customer_churn.csv",
    )

    execution = next(
        item for item in history
        if item["execution_id"] == execution_id
    )

    assert execution["strategies"] == [
        "Pearson correlation",
        "Spearman correlation",
    ]

def test_execution_history_records_strategy_rationale():
    result = AgentResult(
        agent="EDAAgent",
        dataset="customer_churn.csv",
        task="investigate_correlation",
        findings=["x and y have a positive association"],
        features=["x", "y"],
        evidence=[
            Evidence(
                description="Spearman correlation = 0.8",
                method="Spearman correlation",
                value=0.8,
                strategy_reason=(
                    "Strong skew detected; rank-based correlation "
                    "selected as a robustness-oriented alternative."
                ),
            )
        ],
        confidence=0.84,
    )

    kg = KnowledgeGraph()
    engine = SelfEvolutionEngine(kg)

    evaluation = engine.evaluate_execution(
        result,
        {"verification_passed": True},
    )
    engine.update_knowledge(
        result, 
        evaluation, 
        finding_ids=[]
    )

    executions = engine._get_executions(
        task="investigate_correlation",
        dataset="customer_churn.csv",
    )

    assert len(executions) == 1
    assert executions[0]["strategies"] == ["Spearman correlation"]
    assert "Strong skew detected" in executions[0]["strategy_reasons"][0]


def test_strategy_recommendation_uses_successful_history():
    kg = KnowledgeGraph()
    builder = KnowledgeGraphBuilder(kg)
    engine = SelfEvolutionEngine(kg)

    result = create_result(
        agent="EDAAgent",
        task="investigate_correlation",
        finding="Association found",
        evidence_description="Spearman correlation",
        confidence=0.84,
        accuracy=0.87,
        dataset="customer_churn.csv",
    )
    result.evidence[0].method = "Spearman correlation"

    store_execution(
        kg,
        builder,
        engine,
        result,
        verified=True,
    )

    recommendation = engine.recommend_strategy(
        "investigate_correlation",
        dataset="customer_churn.csv",
    )

    assert recommendation["recommended_strategy"] == (
        "Spearman correlation"
    )
def test_strategy_recommendation_does_not_mix_datasets():
    kg = KnowledgeGraph()
    builder = KnowledgeGraphBuilder(kg)
    engine = SelfEvolutionEngine(kg)

    result_a = create_result(
        agent="EDAAgent",
        task="investigate_correlation",
        finding="Finding A",
        evidence_description="Spearman correlation",
        confidence=0.84,
        accuracy=0.87,
        dataset="dataset_a.csv",
    )
    result_a.evidence[0].method = "Spearman correlation"

    store_execution(
        kg,
        builder,
        engine,
        result_a,
        verified=True,
    )

    result_b = create_result(
        agent="EDAAgent",
        task="investigate_correlation",
        finding="Finding B",
        evidence_description="Pearson correlation",
        confidence=0.84,
        accuracy=0.87,
        dataset="dataset_b.csv",
    )
    result_b.evidence[0].method = "Pearson correlation"

    store_execution(
        kg,
        builder,
        engine,
        result_b,
        verified=True,
    )

    recommendation = engine.recommend_strategy(
        "investigate_correlation"
    )

    assert recommendation["recommended_strategy"] is None
    assert "Multiple datasets" in recommendation["reason"]