from app.knowledge_graph.graph import KnowledgeGraph
from app.knowledge_graph.builder import KnowledgeGraphBuilder
from app.schemas import AgentResult, Evidence
from app.verification.fact_checker import FactChecker


def test_model_free_finding():
    # Create knowledge graph and builder.
    kg = KnowledgeGraph()
    builder = KnowledgeGraphBuilder(kg)

    # Create a valid finding with evidence but no ML model.
    result = AgentResult(
        agent="EDAAgent",
        dataset="customer_churn.csv",
        task="correlation_analysis",
        findings=[
            "MonthlyCharges is correlated with churn"
        ],
        evidence=[
            Evidence(
                description="Pearson correlation = 0.42",
                method="Pearson correlation",
                value=0.42,
                finding_indices=[0],
            )
        ],
        model=None,
        confidence=0.84,
    )

    # Add the result to the knowledge graph.
    finding_ids = builder.add_agent_result(result)

    assert len(finding_ids) == 1, (
        "Exactly one finding should be created"
    )

    finding_id = finding_ids[0]

    # Verify the finding's structure.
    checker = FactChecker(kg)
    verification = checker.verify_finding(finding_id)

    print("\nMODEL-FREE VERIFICATION RESULT:")
    print(verification)

    # Scientific support is not assessed by this checker.
    assert verification["verified"] is False

    # The finding can still pass structural validation.
    assert verification["structural_verified"] is True, (
        "A valid finding without an ML model should pass "
        "structural validation"
    )

    assert verification["verification_scope"] == "structural_only"
    assert verification["scientific_support_assessed"] is False

    # Verify the individual structural checks.
    assert verification["checks"]["finding_exists"] is True
    assert verification["checks"]["is_finding"] is True
    assert verification["checks"]["has_confidence"] is True
    assert verification["checks"]["valid_confidence"] is True
    assert verification["checks"]["has_evidence"] is True
    assert verification["checks"]["evidence_connected"] is True
    assert verification["checks"]["valid_evidence_content"] is True

    # An ML model is optional.
    assert verification["checks"]["has_model"] is False
    assert verification["checks"]["valid_model_accuracy"] is True

    print("\nMODEL-FREE TEST PASSED")


if __name__ == "__main__":
    test_model_free_finding()