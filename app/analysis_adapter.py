import math
from typing import Any

from app.confidence_policy import estimate_confidence
from app.schemas import AgentResult, Evidence


def _to_python_value(value: Any) -> Any:
    """Convert NumPy-like scalar values into native Python values."""
    if isinstance(value, dict):
        return {
            key: _to_python_value(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_to_python_value(item) for item in value]

    # NumPy scalars, such as numpy.float64, provide item().
    if hasattr(value, "item") and callable(value.item):
        try:
            return _to_python_value(value.item())
        except (TypeError, ValueError, OverflowError):
            pass

    return value


def _finite_float(value: Any, field_name: str) -> float:
    """Validate a numeric value and return it as a finite float."""
    value = _to_python_value(value)

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite number.")

    try:
        numeric_value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{field_name} must be a finite number."
        ) from exc

    if not math.isfinite(numeric_value):
        raise ValueError(f"{field_name} must be a finite number.")

    return numeric_value


def analysis_result_to_agent_result(
    result: dict,
    dataset: str,
    agent: str = "EDAAgent",
) -> AgentResult:
    """
    Convert a successful analysis result into the shared agent schema.

    Confidence comes from the project's confidence policy; it is
    not the p-value or a calibrated probability.
    """
    if not isinstance(result, dict):
        raise TypeError("Analysis result must be a dictionary.")

    if "error" in result:
        raise ValueError(
            f"Cannot convert a failed analysis result: {result['error']}"
        )

    if not isinstance(dataset, str) or not dataset.strip():
        raise ValueError("Dataset must be a non-empty string.")

    if not isinstance(agent, str) or not agent.strip():
        raise ValueError("Agent must be a non-empty string.")

    task = result.get("task")
    columns = result.get("columns")
    method = result.get("method")
    sample_size = result.get("sample_size")

    if not isinstance(columns, list) or not columns:
        raise ValueError(
            "Analysis result must contain at least one column."
        )

    if not all(
        isinstance(column, str) and column.strip()
        for column in columns
    ):
        raise ValueError("Column names must be non-empty strings.")

    if not isinstance(method, str) or not method.strip():
        raise ValueError("Analysis result must contain a method.")

    if (
        isinstance(sample_size, bool)
        or not isinstance(sample_size, int)
        or sample_size < 0
    ):
        raise ValueError(
            "Analysis result must contain a valid sample_size."
        )

    columns_text = ", ".join(columns)

    if task == "investigate_correlation":
        # Correlation requires exactly two columns.
        if len(columns) != 2:
            raise ValueError(
                "Correlation results must contain exactly two columns."
            )

        if "value" not in result or "p_value" not in result:
            raise ValueError(
                "Correlation result is missing its metrics."
            )

        correlation = _finite_float(
            result["value"],
            "Correlation",
        )

        if not -1.0 <= correlation <= 1.0:
            raise ValueError(
                "Correlation must be finite and between -1 and 1."
            )

        p_value = _finite_float(
            result["p_value"],
            "P-value",
        )

        if not 0.0 <= p_value <= 1.0:
            raise ValueError(
                "P-value must be finite and between 0 and 1."
            )

        finding = (
            f"{method} between {columns[0]} and {columns[1]} "
            f"is {correlation:.3f} (sample size={sample_size})."
        )

    elif task == "investigate_missingness":
        if "n_missing" not in result or "pct_missing" not in result:
            raise ValueError(
                "Missingness result is missing its metrics."
            )

        n_missing = result["n_missing"]
        pct_missing = _finite_float(
            result["pct_missing"],
            "Missing-value percentage",
        )

        if (
            isinstance(n_missing, bool)
            or not isinstance(n_missing, int)
            or not 0 <= n_missing <= sample_size
        ):
            raise ValueError("Missing-value count is invalid.")

        if not 0.0 <= pct_missing <= 100.0:
            raise ValueError(
                "Missing-value percentage must be between 0 and 100."
            )

        finding = (
            f"Column {columns[0]} has {n_missing} missing values "
            f"({pct_missing:.2f}%)."
        )

    elif task == "investigate_outliers":
        if not {
            "n_outliers",
            "pct_outliers",
            "bounds",
        }.issubset(result):
            raise ValueError(
                "Outlier result is missing its metrics."
            )

        n_outliers = result["n_outliers"]
        pct_outliers = _finite_float(
            result["pct_outliers"],
            "Outlier percentage",
        )
        bounds = result["bounds"]

        if (
            isinstance(n_outliers, bool)
            or not isinstance(n_outliers, int)
            or not 0 <= n_outliers <= sample_size
        ):
            raise ValueError("Outlier count is invalid.")

        if not 0.0 <= pct_outliers <= 100.0:
            raise ValueError(
                "Outlier percentage must be between 0 and 100."
            )

        if (
            not isinstance(bounds, (list, tuple))
            or len(bounds) != 2
            or any(
                isinstance(bound, bool)
                or not isinstance(bound, (int, float))
                or not math.isfinite(bound)
                for bound in bounds
            )
        ):
            raise ValueError(
                "Outlier bounds must contain two finite numbers."
            )

        finding = (
            f"Column {columns[0]} has {n_outliers} detected outliers "
            f"({pct_outliers:.2f}%) using the IQR method."
        )

    else:
        raise ValueError(f"Unsupported analysis task: {task!r}")

    evidence = Evidence(
        description=f"{method} evidence for {columns_text}.",
        method=method,
        value=_to_python_value(result),
        finding_indices=[0],
        strategy_reason=result.get("strategy_reason"),
        strategy_context=_to_python_value(result.get("strategy_context")),
    )

    return AgentResult(
        agent=agent,
        dataset=dataset,
        task=task,
        findings=[finding],
        features=columns,
        evidence=[evidence],
        confidence=estimate_confidence(sample_size),
    )