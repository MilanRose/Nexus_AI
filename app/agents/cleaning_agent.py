"""Safe, traceable CSV cleaning for the sequential analytics workflow.

The cleaning stage deliberately does not impute missing values globally.
Missingness remains visible to EDA, while ML preprocessing is fitted inside
its training pipeline to avoid leaking test-set information.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from app.confidence_policy import estimate_confidence
from app.schemas import AgentResult, Evidence
from src.data_loader import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class DataCleaningAgent:
    """Create a cleaned, traceable CSV artifact and explain each change."""

    def run(
        self,
        dataset_path: str,
        output_root: str | Path | None = None,
    ) -> tuple[AgentResult, dict, Path]:
        source_path = Path(dataset_path)
        if not source_path.exists() or not source_path.is_file():
            raise ValueError(f"Dataset does not exist: {source_path}")

        source_bytes = source_path.read_bytes()
        source_sha256 = hashlib.sha256(source_bytes).hexdigest()
        df = load_dataset(str(source_path))
        if df.empty:
            raise ValueError("The dataset contains no rows to clean.")

        input_shape = [int(df.shape[0]), int(df.shape[1])]
        original_columns = [str(column) for column in df.columns]
        normalized_columns = [column.strip() for column in original_columns]
        if len(set(normalized_columns)) != len(normalized_columns):
            raise ValueError(
                "Cleaning column-name whitespace would create duplicate names. "
                "Please make the CSV column names unique before running the workflow."
            )
        renamed_columns = [
            {"from": before, "to": after}
            for before, after in zip(original_columns, normalized_columns)
            if before != after
        ]
        df.columns = normalized_columns

        duplicate_rows_removed = int(df.duplicated().sum())
        if duplicate_rows_removed:
            df = df.drop_duplicates().copy()

        all_empty_columns = [
            str(column) for column in df.columns if df[column].isna().all()
        ]
        if all_empty_columns:
            df = df.drop(columns=all_empty_columns)

        infinite_values_replaced = 0
        numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
        for column in numeric_columns:
            values = df[column]
            infinite_mask = np.isinf(values.to_numpy(dtype=float, copy=False))
            infinite_values_replaced += int(infinite_mask.sum())
            if infinite_mask.any():
                df[column] = values.replace([np.inf, -np.inf], np.nan)

        if df.empty or df.shape[1] == 0:
            raise ValueError("No usable rows or columns remain after cleaning.")

        root = Path(output_root) if output_root else PROJECT_ROOT / "outputs" / "cleaned_datasets"
        cleaned_path = root / source_sha256[:16] / source_path.name
        cleaned_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cleaned_path, index=False)

        details = {
            "agent": "DataCleaningAgent",
            "task": "data_cleaning",
            "source_dataset": source_path.name,
            "cleaned_dataset": source_path.name,
            "source_sha256": source_sha256,
            "input_shape": input_shape,
            "output_shape": [int(df.shape[0]), int(df.shape[1])],
            "duplicate_rows_removed": duplicate_rows_removed,
            "all_empty_columns_removed": all_empty_columns,
            "column_names_trimmed": renamed_columns,
            "infinite_values_converted_to_missing": infinite_values_replaced,
            "missing_values_preserved_for_eda": True,
            "imputation_policy": (
                "No dataset-wide imputation was performed. EDA retains missingness; "
                "the ML preprocessing pipeline fits imputers on training data only."
            ),
            "cleaned_csv_path": str(cleaned_path),
        }

        findings: list[str] = []
        evidence: list[Evidence] = []

        def add_finding(description: str, evidence_description: str, value, method: str) -> None:
            finding_index = len(findings)
            findings.append(description)
            evidence.append(
                Evidence(
                    description=evidence_description,
                    method=method,
                    value=value,
                    finding_indices=[finding_index],
                )
            )

        add_finding(
            f"Created an analysis-ready copy of {source_path.name} with {df.shape[0]} rows and {df.shape[1]} columns.",
            "The cleaning agent wrote a separate CSV artifact and preserved the source file.",
            {"input_shape": input_shape, "output_shape": details["output_shape"], "source_sha256": source_sha256},
            "traceable CSV cleaning",
        )
        add_finding(
            f"Removed {duplicate_rows_removed} exact duplicate row(s).",
            "Exact duplicate rows were counted before being removed.",
            duplicate_rows_removed,
            "exact duplicate removal",
        )
        add_finding(
            f"Removed {len(all_empty_columns)} entirely empty column(s).",
            "Columns with no observed values were removed; partially missing columns were retained.",
            all_empty_columns,
            "all-empty column removal",
        )
        add_finding(
            f"Trimmed whitespace from {len(renamed_columns)} column name(s) and converted {infinite_values_replaced} infinite value(s) to missing values.",
            "Column-name and infinite-value normalization was applied and recorded.",
            {"column_names_trimmed": renamed_columns, "infinite_values_converted_to_missing": infinite_values_replaced},
            "schema and numeric-value normalization",
        )
        add_finding(
            "Missing values were preserved for EDA instead of being imputed globally.",
            "Global imputation can hide missingness and leak information across evaluation splits; ML imputers are fitted on training data only.",
            {"missing_values_preserved": True, "imputation_stage": "training-only ML pipeline"},
            "leakage-aware preprocessing policy",
        )

        result = AgentResult(
            agent="DataCleaningAgent",
            dataset=source_path.name,
            task="data_cleaning",
            findings=findings,
            features=list(df.columns),
            evidence=evidence,
            model=None,
            confidence=estimate_confidence(len(df)),
        )

        manifest_path = cleaned_path.with_suffix(".manifest.json")
        manifest_path.write_text(
            json.dumps(details, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return result, details, cleaned_path
