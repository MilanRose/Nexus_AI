import math

import numpy as np
import pandas as pd

from app.analysis_adapter import analysis_result_to_agent_result
from app.knowledge_graph.builder import KnowledgeGraphBuilder
from app.knowledge_graph.evolution import SelfEvolutionEngine
from app.knowledge_graph.graph import KnowledgeGraph
from app.schemas import AgentResult, Evidence
from src.analysis import build_correlation_context, execute_task


def profile(
    *,
    max_skew,
    mean_skew,
    outliers,
    gap,
    missing=0.0,
    log_size=2.0,
    gap_available=1.0,
    constant_fraction=0.0,
    columns=("x", "y"),
):
    return {
        "version": 1,
        "columns": list(columns),
        "row_count": 100,
        "complete_sample_size": 100,
        "features": {
            "log10_sample_size": log_size,
            "missing_fraction": missing,
            "max_abs_skew": max_skew,
            "mean_abs_skew": mean_skew,
            "max_iqr_outlier_fraction": outliers,
            "pearson_spearman_gap": gap,
            "correlation_gap_available": gap_available,
            "constant_column_fraction": constant_fraction,
        },
    }


def store_strategy(engine, builder, strategy, context, dataset, run_number):
    columns = context.get("columns", ["x", "y"])
    result = AgentResult(
        agent="EDAAgent",
        dataset=dataset,
        task="investigate_correlation",
        findings=[f"{strategy} result for {columns[0]} and {columns[1]}"],
        features=columns,
        evidence=[
            Evidence(
                description=f"{strategy} evidence",
                method=strategy,
                value={"run": run_number},
                finding_indices=[0],
                strategy_reason="Test-only recorded rationale.",
                strategy_context=context,
            )
        ],
        confidence=0.9,
    )
    finding_ids = builder.add_agent_result(result)
    evaluation = engine.evaluate_execution(
        result,
        {"structural_verified": True, "checks": {}},
    )
    engine.update_knowledge(result, evaluation, finding_ids)


def test_correlation_context_is_versioned_finite_and_describes_input_data():
    df = pd.DataFrame({
        "x": [1, 2, 3, 4, 5, 6, np.nan],
        "y": [1, 2, 2, 4, 8, 16, 32],
    })

    context = build_correlation_context(df, "x", "y")

    assert context["version"] == 1
    assert context["columns"] == ["x", "y"]
    assert context["row_count"] == 7
    assert context["complete_sample_size"] == 6
    assert set(context["features"]) >= {
        "log10_sample_size",
        "missing_fraction",
        "max_abs_skew",
        "mean_abs_skew",
        "max_iqr_outlier_fraction",
        "pearson_spearman_gap",
        "correlation_gap_available",
        "constant_column_fraction",
    }
    assert all(math.isfinite(value) for value in context["features"].values())
    assert context["features"]["missing_fraction"] > 0


def test_contextual_retrieval_prefers_similar_profiles_over_global_majority():
    graph = KnowledgeGraph()
    builder = KnowledgeGraphBuilder(graph)
    engine = SelfEvolutionEngine(graph)

    current_context = profile(
        max_skew=3.8,
        mean_skew=3.0,
        outliers=0.25,
        gap=1.2,
        missing=0.20,
        columns=("target_x", "target_y"),
    )

    # More Pearson executions exist globally, but they describe very
    # different profiles and should not dominate contextual retrieval.
    unrelated_pearson_context = profile(
        max_skew=0.0,
        mean_skew=0.0,
        outliers=0.0,
        gap=0.0,
        missing=0.0,
    )
    for index in range(4):
        store_strategy(
            engine,
            builder,
            "Pearson correlation",
            unrelated_pearson_context,
            dataset=f"linear_dataset_{index}.csv",
            run_number=index,
        )

    store_strategy(
        engine,
        builder,
        "Spearman correlation",
        current_context,
        dataset="previous_dataset.csv",
        run_number=10,
    )

    recommendation = engine.recommend_strategy(
        task="investigate_correlation",
        dataset="target_dataset.csv",
        context=current_context,
    )

    assert recommendation["recommended_strategy"] == "Spearman correlation"
    assert recommendation["retrieval_mode"] == "contextual"
    assert recommendation["contextual_matches"] >= 1
    assert recommendation["nearest_similarity"] == 1.0
    assert "contextual memory selected" in recommendation["reason"].lower()
    assert recommendation["supporting_experiences"][0]["dataset"] == "previous_dataset.csv"


def test_contextless_legacy_history_is_not_mistaken_for_contextual_evidence():
    graph = KnowledgeGraph()
    builder = KnowledgeGraphBuilder(graph)
    engine = SelfEvolutionEngine(graph)
    current_context = profile(
        max_skew=2.0,
        mean_skew=1.0,
        outliers=0.10,
        gap=0.5,
    )

    # Legacy result has no strategy context; it remains available to the
    # legacy recommender but must not masquerade as a context match.
    store_strategy(
        engine,
        builder,
        "Spearman correlation",
        {},
        dataset="legacy.csv",
        run_number=1,
    )
    # Empty context is intentionally not stored by update_knowledge; this
    # result should be excluded by the context-aware retrieval path.
    result = engine.recommend_strategy(
        task="investigate_correlation",
        dataset="target.csv",
        context=current_context,
    )

    assert result["recommended_strategy"] is None
    assert result["contextual_matches"] == 0
    assert "no successful historical experiences" in result["reason"].lower()


def test_repeated_runs_of_the_same_context_do_not_dominate_memory():
    graph = KnowledgeGraph()
    builder = KnowledgeGraphBuilder(graph)
    engine = SelfEvolutionEngine(graph)
    same_context = profile(
        max_skew=2.1,
        mean_skew=1.4,
        outliers=0.12,
        gap=0.6,
        columns=("x", "y"),
    )

    for index in range(8):
        store_strategy(
            engine,
            builder,
            "Spearman correlation",
            same_context,
            dataset="repeated_run.csv",
            run_number=index,
        )
    # An alternate strategy observed for the exact same context should make
    # the memory inconclusive; eight reruns must not swamp this one observation.
    store_strategy(
        engine,
        builder,
        "Pearson correlation",
        same_context,
        dataset="repeated_run.csv",
        run_number=99,
    )

    recommendation = engine.recommend_strategy(
        task="investigate_correlation",
        dataset="new_dataset.csv",
        context=same_context,
    )

    assert recommendation["recommended_strategy"] is None
    assert recommendation["contextual_matches"] == 1
    assert "clear strategy preference" in recommendation["reason"].lower()


def test_contextual_strategy_survives_graph_save_and_load(tmp_path):
    graph_path = tmp_path / "strategy-history.json"
    graph = KnowledgeGraph(storage_path=graph_path)
    builder = KnowledgeGraphBuilder(graph)
    engine = SelfEvolutionEngine(graph)

    df = pd.DataFrame({
        "x": np.arange(1, 101, dtype=float),
        "y": np.exp(np.arange(1, 101, dtype=float) / 10),
    })
    task = {
        "task_id": "T001",
        "task": "investigate_correlation",
        "columns": ["x", "y"],
    }

    first_result = execute_task(df, task, dataset="first.csv")
    agent_result = analysis_result_to_agent_result(
        first_result,
        dataset="first.csv",
    )
    finding_ids = builder.add_agent_result(agent_result)
    evaluation = engine.evaluate_execution(
        agent_result,
        {"structural_verified": True, "checks": {}},
    )
    engine.update_knowledge(agent_result, evaluation, finding_ids)
    graph.save()

    reloaded_graph = KnowledgeGraph(storage_path=graph_path)
    reloaded_engine = SelfEvolutionEngine(reloaded_graph)
    second_result = execute_task(
        df,
        task,
        evolution_engine=reloaded_engine,
        dataset="second.csv",
    )

    # Contextual retrieval can transfer a relevant experience across dataset
    # names when their measured profiles match; this is explicit in rationale.
    assert second_result["method"] == first_result["method"]
    assert second_result["strategy_reason"].startswith(
        "Using historical strategy recommendation:"
    )
    assert "contextual memory selected" in second_result["strategy_reason"].lower()
