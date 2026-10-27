import math
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from app.confidence_policy import estimate_confidence
from app.schemas import AgentResult, Evidence, ModelResult
from src.data_loader import load_dataset


class MLAgent:
    """
    Train and evaluate a classification model.

    Version 1 supports classification targets with 2-20 classes.
    It does not yet implement regression or hyperparameter tuning.
    """

    def run(
        self,
        dataset_path: str,
        target_column: str,
        *,
        return_artifacts: bool = False,
    ) -> tuple[AgentResult, dict] | tuple[AgentResult, dict, dict]:
        """
        Train a Random Forest classifier and return structured results.

        Args:
            dataset_path: Path to the CSV dataset.
            target_column: Name of the column to predict.

        Returns:
            A tuple containing an AgentResult and evaluation details.

        Raises:
            ValueError: If the target or dataset is unsuitable.
        """
        path = Path(dataset_path)

        if not isinstance(target_column, str) or not target_column.strip():
            raise ValueError("target_column must be a non-empty string.")

        target_column = target_column.strip()

        df = load_dataset(str(path))

        if target_column not in df.columns:
            raise ValueError(
                f"Target column '{target_column}' was not found. "
                f"Available columns: {list(df.columns)}"
            )

        # Rows without a target cannot be used for supervised learning.
        df = df.loc[df[target_column].notna()].copy()

        if len(df) < 10:
            raise ValueError(
                "At least 10 rows with a non-missing target are required."
            )

        y = df[target_column]
        number_of_classes = int(y.nunique())

        if not 2 <= number_of_classes <= 20:
            raise ValueError(
                "MLAgent version 1 supports classification targets "
                "with 2-20 distinct classes. Regression is not yet "
                "implemented."
            )

        # Prepare candidate features, excluding the target itself.
        X = df.drop(columns=[target_column])

        all_missing_columns = [
            column
            for column in X.columns
            if X[column].isna().all()
        ]

        X = X.drop(columns=all_missing_columns)

        # Exclude common row identifiers and very high-cardinality
        # categorical fields, which can encourage memorization.
        excluded_columns = list(all_missing_columns)

        common_identifier_names = {
            "id",
            "passengerid",
            "customerid",
            "userid",
            "recordid",
            "rowid",
            "index",
        }

        for column in X.columns:
            normalized_name = (
                str(column).strip().lower().replace("_", "").replace(" ", "")
            )

            if normalized_name in common_identifier_names:
                excluded_columns.append(column)

        excluded_columns = list(dict.fromkeys(excluded_columns))
        X = X.drop(
            columns=[
                column
                for column in excluded_columns
                if column in X.columns
            ]
        )

        categorical_candidates = X.select_dtypes(
            include=["object", "category", "bool", "string"]
        ).columns.tolist()

        high_cardinality_columns = [
            column
            for column in categorical_candidates
            if X[column].nunique(dropna=True) > max(
                50,
                math.ceil(len(X) * 0.5),
            )
        ]

        excluded_columns.extend(high_cardinality_columns)
        excluded_columns = list(dict.fromkeys(excluded_columns))

        X = X.drop(
            columns=[
                column
                for column in high_cardinality_columns
                if column in X.columns
            ]
        )

        numeric_columns = X.select_dtypes(
            include=["number"]
        ).columns.tolist()

        categorical_columns = X.select_dtypes(
            include=["object", "category", "bool", "string"]
        ).columns.tolist()

        feature_columns = numeric_columns + categorical_columns

        if not feature_columns:
            raise ValueError(
                "No usable numeric or categorical feature columns remain "
                "after excluding the target and unsuitable columns."
            )

        X = X[feature_columns]

        # Preprocess each feature type without fitting on the test set.
        transformers = []

        if numeric_columns:
            numeric_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                ]
            )

            transformers.append(
                ("numeric", numeric_pipeline, numeric_columns)
            )

        if categorical_columns:
            categorical_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent"),
                    ),
                    (
                        "onehot",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            min_frequency=2,
                        ),
                    ),
                ]
            )

            transformers.append(
                ("categorical", categorical_pipeline, categorical_columns)
            )

        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
        )

        model_pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=100,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

        # Reserve data for final evaluation.
        test_size = max(1, math.ceil(len(y) * 0.2))

        if test_size >= len(y):
            raise ValueError(
                "Not enough rows to create separate training and test sets."
            )

        class_counts = y.value_counts()
        can_stratify = (
            int(class_counts.min()) >= 2
            and test_size >= number_of_classes
            and len(y) - test_size >= number_of_classes
        )

        stratify_values = y if can_stratify else None

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42,
            stratify=stratify_values,
        )

        # Fit preprocessing and model using training data only.
        model_pipeline.fit(X_train, y_train)

        predictions = model_pipeline.predict(X_test)

        accuracy = float(accuracy_score(y_test, predictions))
        precision = float(
            precision_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0,
            )
        )
        recall = float(
            recall_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0,
            )
        )
        f1 = float(
            f1_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0,
            )
        )

        metrics = {
            "accuracy": accuracy,
            "precision_macro": precision,
            "recall_macro": recall,
            "f1_macro": f1,
        }

        model_name = "RandomForestClassifier"

        finding = (
            f"{model_name} achieved {accuracy:.3f} accuracy on "
            f"held-out test data when predicting '{target_column}'."
        )

        # Attach metric-specific evidence to the same finding.
        evidence_items = [
            Evidence(
                description=(
                    f"Held-out test accuracy was {accuracy:.4f} "
                    f"on {len(y_test)} test rows."
                ),
                method="held-out classification evaluation",
                value=accuracy,
                finding_indices=[0],
            ),
            Evidence(
                description=f"Macro precision was {precision:.4f}.",
                method="macro precision",
                value=precision,
                finding_indices=[0],
            ),
            Evidence(
                description=f"Macro recall was {recall:.4f}.",
                method="macro recall",
                value=recall,
                finding_indices=[0],
            ),
            Evidence(
                description=f"Macro F1 score was {f1:.4f}.",
                method="macro F1 score",
                value=f1,
                finding_indices=[0],
            ),
        ]

        result = AgentResult(
            agent="MLAgent",
            dataset=path.name,
            task="classification",
            findings=[finding],
            features=feature_columns,
            evidence=evidence_items,
            model=ModelResult(
                name=model_name,
                metrics=metrics,
            ),
            confidence=estimate_confidence(len(y_test)),
        )

        classifier = model_pipeline.named_steps["classifier"]

        details = {
            "target_column": target_column,
            "model_name": model_name,
            "metrics": metrics,
            "training_samples": int(len(y_train)),
            "test_samples": int(len(y_test)),
            "feature_columns": feature_columns,
            "excluded_columns": excluded_columns,
            "class_labels": [str(value) for value in classifier.classes_],
            "confusion_matrix": confusion_matrix(
                y_test,
                predictions,
                labels=classifier.classes_,
            ).tolist(),
        }

        if return_artifacts:
            # Keep the fitted estimator and held-out partition in memory for
            # the subsequent ExplainabilityAgent stage. These objects are not
            # serialized into graph records or dashboard output.
            artifacts = {
                "model": model_pipeline,
                "X_test": X_test.copy(),
                "y_test": y_test.copy(),
                "target_column": target_column,
                "model_name": model_name,
                "training_samples": int(len(X_train)),
                "test_samples": int(len(X_test)),
                "feature_columns": list(feature_columns),
                "class_labels": [str(value) for value in classifier.classes_],
            }
            return result, details, artifacts

        return result, details
