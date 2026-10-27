from pathlib import Path

from app.confidence_policy import estimate_confidence
from app.schemas import AgentResult, Evidence
from src.data_loader import load_dataset
from src.profiler import profile_dataset


class DataAgent:
    """
    Inspect a dataset and report data-quality findings.

    This agent reuses the existing dataset loader and profiler.
    It does not modify the original dataset.
    """

    def run(self, dataset_path: str) -> tuple[AgentResult, dict]:
        """
        Profile a CSV and return an AgentResult plus the full profile.

        Findings cover dataset dimensions, missing values, and
        duplicate rows. Each finding has associated evidence.
        """
        path = Path(dataset_path)

        # Use the existing loader with its resource limits.
        df = load_dataset(str(path))
        profile = profile_dataset(df)

        findings = []
        evidence_items = []

        def add_finding(
            description: str,
            evidence_description: str,
            value,
            method: str,
        ) -> None:
            finding_index = len(findings)

            findings.append(description)

            evidence_items.append(
                Evidence(
                    description=evidence_description,
                    method=method,
                    value=value,
                    finding_indices=[finding_index],
                )
            )

        row_count = int(profile["n_rows"])
        column_count = int(profile["n_cols"])

        # Finding 1: dataset dimensions.
        add_finding(
            description=(
                f"Dataset contains {row_count} rows "
                f"and {column_count} columns."
            ),
            evidence_description=(
                f"Dataset profiler measured {row_count} rows "
                f"and {column_count} columns."
            ),
            value=row_count,
            method="dataset profiling",
        )

        # Findings for columns containing missing values.
        missing_values = profile["missing_values"]
        missing_percentages = profile["missing_pct"]

        columns_with_missing_values = []

        for column in profile["columns"]:
            missing_count = int(missing_values[column])

            if missing_count == 0:
                continue

            missing_percentage = (
                float(missing_percentages[column])
                if row_count > 0
                else 0.0
            )

            columns_with_missing_values.append(column)

            add_finding(
                description=(
                    f"Column {column} has {missing_count} "
                    f"missing values ({missing_percentage:.2f}%)."
                ),
                evidence_description=(
                    f"Dataset profiler counted {missing_count} "
                    f"missing values in column {column}, "
                    f"representing {missing_percentage:.2f}% "
                    f"of {row_count} rows."
                ),
                value=missing_count,
                method="missing-value profiling",
            )

        if not columns_with_missing_values:
            add_finding(
                description="No missing values were detected.",
                evidence_description=(
                    f"Dataset profiler found zero missing values "
                    f"across {column_count} columns."
                ),
                value=0,
                method="missing-value profiling",
            )

        # Finding: duplicate rows.
        duplicate_count = int(profile["duplicate_rows"])

        if duplicate_count:
            duplicate_description = (
                f"Dataset contains {duplicate_count} duplicate rows."
            )
        else:
            duplicate_description = (
                "No duplicate rows were detected."
            )

        add_finding(
            description=duplicate_description,
            evidence_description=(
                f"Dataset profiler counted {duplicate_count} "
                f"duplicate rows in the loaded dataset."
            ),
            value=duplicate_count,
            method="duplicate-row profiling",
        )

        # Use the existing provisional sample-size heuristic.
        confidence = estimate_confidence(row_count)

        result = AgentResult(
            agent="DataAgent",
            dataset=path.name,
            task="data_quality_profile",
            findings=findings,
            features=list(profile["columns"]),
            evidence=evidence_items,
            model=None,
            confidence=confidence,
        )

        return result, profile