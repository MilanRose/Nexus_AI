
"""Explain classification predictions using permutation importance.

This agent fits a reference Random Forest model and measures how much
held-out accuracy changes when one input feature is shuffled.

Important: this is a newly fitted reference model. It does not claim
to explain a fitted model instance previously created by MLAgent.
"""

from pathlib import Path
import math

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from app.confidence_policy import estimate_confidence
from app.schemas import AgentResult, Evidence, ModelResult
from src.data_loader import load_dataset


class ExplainabilityAgent:
    """Explain a reference classification model's feature contributions."""

    AGENT_NAME = "ExplainabilityAgent"
    MODEL_NAME = "RandomForestClassifier"

    def explain_fitted_model(
        self,
        model_artifacts,
        *,
        dataset_name,
        target_column,
        top_n=10,
        n_repeats=5,
    ):
        """Explain the exact fitted MLAgent model on its held-out partition.

        `model_artifacts` is an in-memory handoff from MLAgent. The fitted
        estimator and test partition are never written into the knowledge
        graph or serialized to the dashboard.
        """
        if not isinstance(model_artifacts, dict):
            raise TypeError("model_artifacts must be a dictionary.")
        model = model_artifacts.get("model")
        X_test = model_artifacts.get("X_test")
        y_test = model_artifacts.get("y_test")
        if model is None or not isinstance(X_test, pd.DataFrame) or y_test is None:
            raise ValueError(
                "Fitted-model artifacts must contain model, DataFrame X_test, and y_test."
            )
        if X_test.empty or len(X_test) != len(y_test):
            raise ValueError("The held-out explanation dataset is empty or misaligned.")
        if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n < 1:
            raise ValueError("top_n must be a positive integer.")
        if not isinstance(n_repeats, int) or isinstance(n_repeats, bool) or not 1 <= n_repeats <= 30:
            raise ValueError("n_repeats must be an integer between 1 and 30.")

        predictions = model.predict(X_test)
        baseline_accuracy = float(accuracy_score(y_test, predictions))
        macro_f1 = float(
            f1_score(y_test, predictions, average="macro", zero_division=0)
        )
        importance_result = permutation_importance(
            model,
            X_test,
            y_test,
            scoring="accuracy",
            n_repeats=n_repeats,
            random_state=42,
            n_jobs=1,
        )
        feature_records = []
        for index, column in enumerate(X_test.columns):
            feature_records.append({
                "feature": str(column),
                "mean_accuracy_decrease": float(importance_result.importances_mean[index]),
                "standard_deviation": float(importance_result.importances_std[index]),
            })
        feature_records.sort(
            key=lambda record: record["mean_accuracy_decrease"],
            reverse=True,
        )
        top_features = feature_records[:top_n]

        findings = []
        evidence = []
        for index, record in enumerate(top_features):
            feature = record["feature"]
            importance = record["mean_accuracy_decrease"]
            std = record["standard_deviation"]
            findings.append(
                f"Permutation importance for feature '{feature}': mean held-out "
                f"accuracy decrease = {importance:.4f} (standard deviation = {std:.4f})."
            )
            evidence.append(Evidence(
                description=f"Held-out permutation importance for feature '{feature}'",
                method="Permutation importance on the MLAgent fitted model",
                value={
                    "feature": feature,
                    "mean_accuracy_decrease": importance,
                    "standard_deviation": std,
                    "repeats": n_repeats,
                    "baseline_accuracy": baseline_accuracy,
                    "model_source": "exact_fitted_model_from_MLAgent",
                },
                finding_indices=[index],
            ))

        model_name = model_artifacts.get("model_name", self.MODEL_NAME)
        result = AgentResult(
            agent=self.AGENT_NAME,
            dataset=str(dataset_name),
            task="permutation_feature_importance",
            findings=findings,
            features=[record["feature"] for record in top_features],
            evidence=evidence,
            model=ModelResult(
                name=str(model_name),
                metrics={"accuracy": baseline_accuracy, "f1_macro": macro_f1},
            ),
            confidence=estimate_confidence(len(X_test)),
        )
        details = {
            "agent": self.AGENT_NAME,
            "task": "permutation_feature_importance",
            "dataset": str(dataset_name),
            "target_column": str(target_column),
            "model_name": str(model_name),
            "model_source": "exact_fitted_model_from_MLAgent",
            "explanation_method": "permutation_importance",
            "evaluation_method": "accuracy decrease on the same held-out test partition used by MLAgent",
            "baseline_accuracy": baseline_accuracy,
            "f1_macro": macro_f1,
            "train_samples": int(model_artifacts.get("training_samples", 0)),
            "test_samples": int(len(X_test)),
            "class_count": int(pd.Series(y_test).nunique()),
            "classes": [str(value) for value in pd.Series(y_test).drop_duplicates().tolist()],
            "feature_columns": [str(column) for column in X_test.columns],
            "feature_importance": top_features,
            "limitations": [
                "This explanation uses the exact fitted model returned by MLAgent and the same held-out test partition.",
                "Permutation importance measures predictive dependence under this evaluation split; it does not establish causality.",
                "Correlated features can share or mask each other's importance.",
                "Negative importance means shuffling the feature improved accuracy on average in these permutation trials; it does not prove the feature is universally harmful.",
            ],
        }
        return result, details

    def run(
        self,
        dataset_path,
        target_column,
        top_n=10,
        n_repeats=5,
    ):
        """
        Fit a reference classifier and calculate permutation importance.

        Returns:
            (AgentResult, details)

        Each importance score is the mean decrease in held-out accuracy
        after randomly permuting one original input column. A negative
        score means shuffling that column improved accuracy on average.
        """

        if (
            not isinstance(target_column, str)
            or not target_column.strip()
        ):
            raise ValueError(
                "target_column must be a non-empty column name."
            )

        target_column = target_column.strip()

        if (
            not isinstance(top_n, int)
            or isinstance(top_n, bool)
            or top_n < 1
        ):
            raise ValueError(
                "top_n must be a positive integer."
            )

        if (
            not isinstance(n_repeats, int)
            or isinstance(n_repeats, bool)
            or n_repeats < 1
            or n_repeats > 30
        ):
            raise ValueError(
                "n_repeats must be an integer between 1 and 30."
            )

        dataset_path = Path(dataset_path)

        if not dataset_path.exists():
            raise ValueError(
                f"Dataset does not exist: {dataset_path}"
            )

        df = load_dataset(str(dataset_path))

        if not isinstance(df, pd.DataFrame):
            raise ValueError(
                "The dataset loader did not return a DataFrame."
            )

        if df.empty:
            raise ValueError("The dataset contains no rows.")

        if target_column not in df.columns:
            raise ValueError(
                f"Target column '{target_column}' was not found. "
                f"Available columns: {list(df.columns)}"
            )

        # Remove rows where the requested target is unknown.
        df = df.loc[df[target_column].notna()].copy()

        if len(df) < 10:
            raise ValueError(
                "At least 10 rows with non-missing target values "
                "are required for explanation."
            )

        class_count = int(df[target_column].nunique(dropna=True))

        if class_count < 2 or class_count > 20:
            raise ValueError(
                "ExplainabilityAgent currently supports classification "
                "targets with 2 to 20 distinct classes. "
                f"Found {class_count}."
            )

        y = df[target_column].copy()
        X = df.drop(columns=[target_column]).copy()

        # Remove columns that contain no observed data.
        all_missing_columns = [
            column
            for column in X.columns
            if X[column].isna().all()
        ]

        if all_missing_columns:
            X = X.drop(columns=all_missing_columns)

        # Exclude common identifier columns to reduce the risk that
        # arbitrary row IDs dominate feature importance.
        identifier_names = {
            "id",
            "index",
            "rowid",
            "recordid",
            "passengerid",
            "customerid",
            "userid",
            "user_id",
            "customer_id",
            "record_id",
            "transaction_id",
            "uuid",
        }

        excluded_columns = list(all_missing_columns)

        identifier_columns = []

        for column in X.columns:
            normalized = (
                str(column)
                .strip()
                .lower()
                .replace(" ", "")
                .replace("-", "")
            )

            if normalized in identifier_names:
                identifier_columns.append(column)

        if identifier_columns:
            X = X.drop(columns=identifier_columns)
            excluded_columns.extend(identifier_columns)

        # Remove categorical columns with unusually high cardinality.
        # This helps prevent name-like or nearly unique fields from
        # becoming enormous one-hot encoded feature spaces.
        high_cardinality_columns = []
        cardinality_limit = max(
            50,
            math.ceil(len(X) * 0.5),
        )

        for column in X.columns:
            is_categorical = (
                not pd.api.types.is_numeric_dtype(X[column])
                and not pd.api.types.is_bool_dtype(X[column])
            )

            if (
                is_categorical
                and X[column].nunique(dropna=True)
                > cardinality_limit
            ):
                high_cardinality_columns.append(column)

        if high_cardinality_columns:
            X = X.drop(columns=high_cardinality_columns)
            excluded_columns.extend(high_cardinality_columns)

        # Replace infinite numeric values with missing values so that
        # the numeric imputer can handle them.
        numeric_columns = X.select_dtypes(
            include=["number", "bool"]
        ).columns.tolist()

        for column in numeric_columns:
            X[column] = X[column].replace(
                [np.inf, -np.inf],
                np.nan,
            )

        # Remove columns that became entirely missing after cleaning.
        newly_missing_columns = [
            column
            for column in X.columns
            if X[column].isna().all()
        ]

        if newly_missing_columns:
            X = X.drop(columns=newly_missing_columns)
            excluded_columns.extend(newly_missing_columns)

        if X.empty or len(X.columns) == 0:
            raise ValueError(
                "No usable feature columns remain after excluding "
                "identifiers and unusable columns."
            )

        numeric_columns = X.select_dtypes(
            include=["number", "bool"]
        ).columns.tolist()

        categorical_columns = [
            column
            for column in X.columns
            if column not in numeric_columns
        ]

        # Keep categorical values consistent while preserving missing
        # values for the imputer.
        for column in categorical_columns:
            X[column] = (
                X[column]
                .where(X[column].notna(), np.nan)
                .astype(object)
            )

        transformers = []

        if numeric_columns:
            numeric_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="median"),
                    )
                ]
            )

            transformers.append(
                (
                    "numeric",
                    numeric_pipeline,
                    numeric_columns,
                )
            )

        if categorical_columns:
            categorical_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="most_frequent"
                        ),
                    ),
                    (
                        "onehot",
                        OneHotEncoder(
                            handle_unknown="ignore"
                        ),
                    ),
                ]
            )

            transformers.append(
                (
                    "categorical",
                    categorical_pipeline,
                    categorical_columns,
                )
            )

        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
        )

        model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=100,
                        random_state=42,
                        n_jobs=1,
                    ),
                ),
            ]
        )

        # Use stratification when every class has enough examples and
        # both partitions can contain every class.
        class_counts = y.value_counts()
        test_count = math.ceil(len(y) * 0.2)
        train_count = len(y) - test_count

        stratify = None

        if (
            int(class_counts.min()) >= 2
            and test_count >= class_count
            and train_count >= class_count
        ):
            stratify = y

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=stratify,
        )

        if y_train.nunique() < 2:
            raise ValueError(
                "The training split contains fewer than two classes. "
                "More representative data is required."
            )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        baseline_accuracy = float(
            accuracy_score(y_test, predictions)
        )

        macro_f1 = float(
            f1_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0,
            )
        )

        # Permute original columns, rather than one-hot encoded columns.
        # This produces an easier-to-interpret score for each source field.
        importance_result = permutation_importance(
            model,
            X_test,
            y_test,
            scoring="accuracy",
            n_repeats=n_repeats,
            random_state=42,
            n_jobs=1,
        )

        feature_records = []

        for index, column in enumerate(X.columns):
            mean_importance = float(
                importance_result.importances_mean[index]
            )

            std_importance = float(
                importance_result.importances_std[index]
            )

            feature_records.append(
                {
                    "feature": str(column),
                    "mean_accuracy_decrease": mean_importance,
                    "standard_deviation": std_importance,
                }
            )

        feature_records.sort(
            key=lambda record: record[
                "mean_accuracy_decrease"
            ],
            reverse=True,
        )

        top_features = feature_records[:top_n]

        findings = []
        evidence = []

        for index, record in enumerate(top_features):
            feature = record["feature"]
            importance = record["mean_accuracy_decrease"]
            standard_deviation = record["standard_deviation"]

            findings.append(
                (
                    f"Permutation importance for feature '{feature}': "
                    f"mean held-out accuracy decrease = "
                    f"{importance:.4f} "
                    f"(standard deviation = {standard_deviation:.4f})."
                )
            )

            evidence.append(
                Evidence(
                    description=(
                        "Held-out permutation importance for "
                        f"feature '{feature}'"
                    ),
                    method=(
                        "Permutation importance using held-out "
                        "classification accuracy"
                    ),
                    value={
                        "feature": feature,
                        "mean_accuracy_decrease": importance,
                        "standard_deviation": standard_deviation,
                        "repeats": n_repeats,
                        "baseline_accuracy": baseline_accuracy,
                    },
                    finding_indices=[index],
                )
            )

        agent_result = AgentResult(
            agent=self.AGENT_NAME,
            dataset=dataset_path.name,
            task="permutation_feature_importance",
            findings=findings,
            features=[
                record["feature"]
                for record in top_features
            ],
            evidence=evidence,
            model=ModelResult(
                name=self.MODEL_NAME,
                metrics={
                    "accuracy": baseline_accuracy,
                    "f1_macro": macro_f1,
                },
            ),
            confidence=estimate_confidence(len(df)),
        )

        details = {
            "agent": self.AGENT_NAME,
            "task": "permutation_feature_importance",
            "dataset": dataset_path.name,
            "target_column": target_column,
            "model_name": self.MODEL_NAME,
            "model_source": "refitted_reference_model",
            "explanation_method": "permutation_importance",
            "evaluation_method": (
                "accuracy measured on held-out test data"
            ),
            "baseline_accuracy": baseline_accuracy,
            "f1_macro": macro_f1,
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "class_count": class_count,
            "classes": [
                str(value)
                for value in pd.unique(y)
            ],
            "feature_columns": [
                str(column)
                for column in X.columns
            ],
            "excluded_columns": list(
                dict.fromkeys(
                    str(column)
                    for column in excluded_columns
                )
            ),
            "feature_importance": top_features,
            "limitations": [
                (
                    "This explanation belongs to a newly fitted "
                    "reference Random Forest model, not necessarily "
                    "the exact model instance produced by MLAgent."
                ),
                (
                    "Permutation importance measures predictive "
                    "dependence under the chosen test split; it does "
                    "not establish causality."
                ),
                (
                    "Correlated features can share or mask each "
                    "other's importance."
                ),
                (
                    "Negative importance means that shuffling a "
                    "feature improved accuracy on average in these "
                    "permutation trials; it does not prove the "
                    "feature is universally harmful."
                ),
            ],
        }

        return agent_result, details
