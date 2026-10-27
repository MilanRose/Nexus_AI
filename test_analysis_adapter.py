import pytest

from app.analysis_adapter import analysis_result_to_agent_result
from app.knowledge_graph.builder import KnowledgeGraphBuilder
from app.knowledge_graph.graph import KnowledgeGraph
from app.verification.fact_checker import FactChecker


def test_correlation_result_reaches_knowledge_graph_and_fact_checker():
    raw_result = {
        "task_id": "T001",
        "task": "investigate_correlation",
        "columns": ["Pclass", "Fare"],
        "method": "Pearson correlation",
        "value": -0.549,
        "p_value": 0.0,
        "sample_size": 891,
    }

    agent_result = analysis_result_to_agent_result(
        raw_result,
        dataset="titanic.csv",
    )

    assert agent_result.confidence == 0.9
    assert agent_result.findings == [
        "Pearson correlation between Pclass and Fare "
        "is -0.549 (sample size=891)."
    ]
    assert agent_result.evidence[0].finding_indices == [0]
    assert agent_result.features == ["Pclass", "Fare"]

    kg = KnowledgeGraph()
    builder = KnowledgeGraphBuilder(kg)
    fact_checker = FactChecker(kg)

    finding_ids = builder.add_agent_result(agent_result)
    assert len(finding_ids) == 1

    verification = fact_checker.verify_finding(finding_ids[0])

    assert verification["verified"] is False
    assert verification["structural_verified"] is True
    assert verification["checks"]["has_evidence"] is True
    assert verification["checks"]["evidence_connected"] is True


def test_missingness_result_reaches_knowledge_graph_and_fact_checker():
    raw_result = {
        "task_id": "T001",
        "task": "investigate_missingness",
        "columns": ["Cabin"],
        "method": "Missing value count",
        "n_missing": 687,
        "pct_missing": 77.1,
        "sample_size": 891,
    }

    agent_result = analysis_result_to_agent_result(
        raw_result,
        dataset="titanic.csv",
    )

    assert agent_result.findings == [
        "Column Cabin has 687 missing values (77.10%)."
    ]

    kg = KnowledgeGraph()
    finding_ids = KnowledgeGraphBuilder(kg).add_agent_result(agent_result)
    verification = FactChecker(kg).verify_finding(finding_ids[0])

    assert verification["verified"] is False
    assert verification["structural_verified"] is True
    assert verification["checks"]["evidence_connected"] is True


def test_outlier_result_reaches_knowledge_graph_and_fact_checker():
    raw_result = {
        "task_id": "T002",
        "task": "investigate_outliers",
        "columns": ["Fare"],
        "method": "IQR outlier detection",
        "n_outliers": 116,
        "pct_outliers": 13.02,
        "bounds": [-26.72, 65.63],
        "sample_size": 891,
    }

    agent_result = analysis_result_to_agent_result(
        raw_result,
        dataset="titanic.csv",
    )

    assert agent_result.findings == [
        "Column Fare has 116 detected outliers "
        "(13.02%) using the IQR method."
    ]

    kg = KnowledgeGraph()
    finding_ids = KnowledgeGraphBuilder(kg).add_agent_result(agent_result)
    verification = FactChecker(kg).verify_finding(finding_ids[0])

    assert verification["verified"] is False
    assert verification["structural_verified"] is True
    assert verification["checks"]["evidence_connected"] is True


@pytest.mark.parametrize(
    "p_value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
        -0.01,
        1.01,
        "not-a-number",
        True,
    ],
)
def test_correlation_rejects_invalid_p_values(p_value):
    raw_result = {
        "task_id": "T003",
        "task": "investigate_correlation",
        "columns": ["Pclass", "Fare"],
        "method": "Pearson correlation",
        "value": -0.549,
        "p_value": p_value,
        "sample_size": 891,
    }

    with pytest.raises(ValueError, match="P-value"):
        analysis_result_to_agent_result(
            raw_result,
            dataset="titanic.csv",
        )


@pytest.mark.parametrize(
    "columns",
    [
        ["Fare"],
        ["Pclass", "Fare", "Age"],
    ],
)
def test_correlation_requires_exactly_two_columns(columns):
    raw_result = {
        "task_id": "T004",
        "task": "investigate_correlation",
        "columns": columns,
        "method": "Pearson correlation",
        "value": -0.549,
        "p_value": 0.01,
        "sample_size": 891,
    }

    with pytest.raises(
        ValueError,
        match="exactly two columns",
    ):
        analysis_result_to_agent_result(
            raw_result,
            dataset="titanic.csv",
        )