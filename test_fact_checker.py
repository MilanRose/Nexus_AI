
from app.schemas import AgentResult, Evidence, ModelResult
from app.knowledge_graph.graph import KnowledgeGraph
from app.knowledge_graph.builder import KnowledgeGraphBuilder
from app.verification.fact_checker import FactChecker


def test_fact_checker():
    kg = KnowledgeGraph()
    builder = KnowledgeGraphBuilder(kg)
    fact_checker = FactChecker(kg)

    result = AgentResult(
        agent="EDAAgent",
        dataset="customer_churn.csv",
        task="correlation_analysis",
        findings=["MonthlyCharges is associated with churn"],
        features=["MonthlyCharges"],
        evidence=[
            Evidence(
                description="Pearson correlation = 0.42",
                method="Pearson correlation",
                value=0.42,
                finding_indices=[0]
            )
        ],
        model=ModelResult(
            name="Random Forest",
            metrics={"accuracy": 0.87}
        ),
        confidence=0.84
    )

    finding_ids = builder.add_agent_result(result)
    assert finding_ids

    verification = fact_checker.verify_finding(finding_ids[0])

    assert verification["verified"] is False
    assert verification["structural_verified"] is True
    assert verification["checks"]["has_evidence"] is True
    assert verification["checks"]["evidence_connected"] is True
    assert verification["checks"]["has_model"] is True


def test_fact_checker_rejects_non_finding_node():
    kg = KnowledgeGraph()
    kg.add_node("dataset_1", "dataset", name="customer_churn.csv")

    verification = FactChecker(kg).verify_finding("dataset_1")

    assert verification["verified"] is False
    assert verification["checks"]["finding_exists"] is True
    assert verification["checks"]["is_finding"] is False


def test_fact_checker_rejects_nan_confidence():
    kg = KnowledgeGraph()
    kg.add_node("finding_nan", "finding", confidence=float("nan"))
    kg.add_node("evidence_1", "evidence", description="Test evidence")
    kg.add_relationship("finding_nan", "supported_by", "evidence_1")

    verification = FactChecker(kg).verify_finding("finding_nan")

    assert verification["verified"] is False
    assert verification["checks"]["valid_confidence"] is False


def test_fact_checker_rejects_invalid_model_accuracy():
    kg = KnowledgeGraph()
    kg.add_node("finding_1", "finding", confidence=0.9)
    kg.add_node("evidence_1", "evidence", description="Test evidence")
    kg.add_node("model_1", "model", accuracy=float("nan"))

    kg.add_relationship("finding_1", "supported_by", "evidence_1")
    kg.add_relationship("finding_1", "generated_by", "model_1")

    verification = FactChecker(kg).verify_finding("finding_1")

    assert verification["verified"] is False
    assert verification["checks"]["valid_model_accuracy"] is False


def test_fact_checker_rejects_out_of_range_confidence():
    kg = KnowledgeGraph()
    kg.add_node("finding_invalid", "finding", confidence=1.5)
    kg.add_node("evidence_1", "evidence", description="Test evidence")
    kg.add_relationship("finding_invalid", "supported_by", "evidence_1")

    verification = FactChecker(kg).verify_finding("finding_invalid")

    assert verification["verified"] is False
    assert verification["checks"]["valid_confidence"] is False


def test_fact_checker_rejects_extremely_large_integer_confidence():
    kg = KnowledgeGraph()
    kg.add_node("finding_large", "finding", confidence=10**1000)
    kg.add_node("evidence_1", "evidence", description="Test evidence")
    kg.add_relationship("finding_large", "supported_by", "evidence_1")

    verification = FactChecker(kg).verify_finding("finding_large")

    assert verification["verified"] is False
    assert verification["checks"]["valid_confidence"] is False


def test_fact_checker_rejects_extremely_large_integer_model_accuracy():
    kg = KnowledgeGraph()
    kg.add_node("finding_1", "finding", confidence=0.9)
    kg.add_node("evidence_1", "evidence", description="Test evidence")
    kg.add_node("model_large", "model", accuracy=10**1000)

    kg.add_relationship("finding_1", "supported_by", "evidence_1")
    kg.add_relationship("finding_1", "generated_by", "model_large")

    verification = FactChecker(kg).verify_finding("finding_1")

    assert verification["verified"] is False
    assert verification["checks"]["valid_model_accuracy"] is False

def test_fact_checker_explicitly_reports_structural_only_verification():
    kg = KnowledgeGraph()
    builder = KnowledgeGraphBuilder(kg)

    result = AgentResult(
        agent="EDAAgent",
        dataset="customer_churn.csv",
        task="correlation_analysis",
        findings=["MonthlyCharges is associated with churn"],
        features=["MonthlyCharges"],
        evidence=[
            Evidence(
                description="Pearson correlation = 0.42",
                method="Pearson correlation",
                value=0.42,
                finding_indices=[0],
            )
        ],
        confidence=0.84,
    )

    finding_ids = builder.add_agent_result(result)
    verification = FactChecker(kg).verify_finding(finding_ids[0])

    assert verification["verified"] is False
    assert verification["structural_verified"] is True
    assert verification["verification_scope"] == "structural_only"
    assert verification["scientific_support_assessed"] is False
    assert "not been assessed" in verification["reason"]


def test_fact_checker_rejects_empty_evidence_description():
    kg = KnowledgeGraph()

    kg.add_node(
        "finding_1",
        "finding",
        description="A claim",
        confidence=0.9,
    )
    kg.add_node(
        "evidence_1",
        "evidence",
        description="   ",
    )
    kg.add_relationship(
        "finding_1",
        "supported_by",
        "evidence_1",
    )

    verification = FactChecker(kg).verify_finding("finding_1")

    assert verification["verified"] is False
    assert verification["checks"]["evidence_connected"] is True
    assert verification["checks"]["valid_evidence_content"] is False


def test_fact_checker_does_not_claim_to_assess_scientific_truth():
    kg = KnowledgeGraph()

    # Deliberately use contradictory text. The structural checker
    # cannot determine whether the description supports the finding.
    kg.add_node(
        "finding_1",
        "finding",
        description="The treatment is effective",
        confidence=0.9,
    )
    kg.add_node(
        "evidence_1",
        "evidence",
        description="The study found no treatment effect.",
    )
    kg.add_relationship(
        "finding_1",
        "supported_by",
        "evidence_1",
    )

    verification = FactChecker(kg).verify_finding("finding_1")

    assert verification["structural_verified"] is True
    assert verification["scientific_support_assessed"] is False
    assert verification["verification_scope"] == "structural_only"
    assert verification["verified"] is False