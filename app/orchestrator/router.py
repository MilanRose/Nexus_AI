
"""Keyword-based routing for the framework's analysis agents."""


class AdaptiveRouter:
    """Select an agent based on the user's request."""

    def route(self, task):
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task must be a non-empty string.")

        task = task.lower().strip()

        # Specific model-explanation requests must be matched before
        # the generic "explain" and "model" keywords below.
        explainability_keywords = [
            "explainability",
            "feature importance",
            "permutation importance",
            "explainable ai",
            "model explanation",
            "explain the model",
            "explain model",
            "interpret model",
            "model interpretability",
            "why did the model",
            "why does the model",
            "why is the model",
            "explain prediction",
            "explain predictions",
            "explain the prediction",
            "explain the predictions",
        ]

        if any(
            phrase in task
            for phrase in explainability_keywords
        ) or "shap" in task.split():
            return "ExplainabilityAgent"

        # Data quality and preparation.
        if any(
            word in task
            for word in [
                "clean",
                "missing values",
                "preprocess",
            ]
        ):
            return "DataAgent"

        # Exploratory data analysis.
        if any(
            word in task
            for word in [
                "explore",
                "eda",
                "correlation",
                "pattern",
            ]
        ):
            return "EDAAgent"

        # Predictive modeling.
        if any(
            word in task
            for word in [
                "predict",
                "classification",
                "regression",
                "model",
            ]
        ):
            return "MLAgent"

        # Report generation from stored findings.
        if any(
            word in task
            for word in [
                "report",
                "explain",
            ]
        ):
            return "ReportAgent"

        # Preserve the existing default route.
        return "DataAgent"
