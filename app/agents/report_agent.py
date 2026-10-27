
"""Generate evidence-aware reports from the existing knowledge graph.

ReportAgent does not run new analyses. It summarizes findings, evidence,
model metadata, and execution records already stored in the graph.
"""

import json
from datetime import datetime, timezone

from app.knowledge_graph.graph import GRAPH_LOCK


class ReportAgent:
    """Create structured and Markdown reports from graph data."""

    def run(
        self,
        knowledge_graph,
        dataset_name=None,
        title=None,
    ):
        """
        Generate a report from a KnowledgeGraph or NetworkX graph.

        Args:
            knowledge_graph: KnowledgeGraph wrapper or NetworkX graph.
            dataset_name: Optional exact dataset name to report on.
            title: Optional report title.

        Returns:
            A dictionary containing the report, its findings, linked
            evidence, model metadata, and execution-history summary.
        """

        graph = self._resolve_graph(knowledge_graph)

        # Copy graph data under the shared lock so the report uses
        # a consistent snapshot while other components may be running.
        with GRAPH_LOCK:
            node_data = {
                node_id: dict(attributes)
                for node_id, attributes in graph.nodes(data=True)
            }

            relations = {}

            if graph.is_multigraph():
                edges = graph.edges(
                    keys=True,
                    data=True,
                )

                for source, target, key, attributes in edges:
                    relation = attributes.get("relation", key)
                    relations.setdefault(
                        (source, relation), []
                    ).append(target)
            else:
                for source, target, attributes in graph.edges(
                    data=True
                ):
                    relation = attributes.get("relation")
                    if relation:
                        relations.setdefault(
                            (source, relation), []
                        ).append(target)

        datasets = self._select_datasets(
            node_data=node_data,
            relations=relations,
            dataset_name=dataset_name,
        )

        dataset_reports = []

        for dataset_id, label in datasets:
            finding_ids = self._findings_for_dataset(
                node_data=node_data,
                relations=relations,
                dataset_id=dataset_id,
                dataset_name=label,
                all_findings=(
                    dataset_id is None
                    and dataset_name is None
                ),
            )

            finding_records = [
                self._build_finding_record(
                    finding_id=finding_id,
                    node_data=node_data,
                    relations=relations,
                    dataset_label=label,
                )
                for finding_id in finding_ids
                if node_data.get(finding_id, {}).get("type")
                == "finding"
            ]

            dataset_reports.append(
                {
                    "dataset": label,
                    "finding_count": len(finding_records),
                    "findings": finding_records,
                }
            )

        all_findings = [
            finding
            for dataset in dataset_reports
            for finding in dataset["findings"]
        ]

        finding_ids_in_report = {
            finding["finding_id"]
            for finding in all_findings
        }

        execution_history = self._get_execution_history(
            node_data=node_data,
            relations=relations,
            finding_ids=finding_ids_in_report,
        )

        evidence_count = sum(
            bool(finding["evidence"])
            for finding in all_findings
        )

        model_count = sum(
            bool(finding["models"])
            for finding in all_findings
        )

        if all_findings:
            summary = (
                f"The knowledge graph contains "
                f"{len(all_findings)} recorded finding(s) across "
                f"{len(dataset_reports)} dataset(s). "
                f"{evidence_count} finding(s) have linked evidence "
                f"records, and {model_count} finding(s) have linked "
                f"model metadata."
            )
            status = "completed"
        else:
            summary = (
                "No findings were found in the knowledge graph "
                "for the selected dataset scope."
            )
            status = "no_findings"

        limitations = [
            (
                "This report summarizes information already stored "
                "in the knowledge graph; it does not rerun analyses."
            ),
            (
                "Linked evidence indicates recorded provenance. "
                "It does not by itself prove that a conclusion is true."
            ),
            (
                "Structural verification is not scientific verification. "
                "Model confidence fields, where present, should not be "
                "interpreted as calibrated probabilities."
            ),
        ]

        report_title = (
            title.strip()
            if isinstance(title, str) and title.strip()
            else "Data Analytics Report"
        )

        generated_at = datetime.now(
            timezone.utc
        ).isoformat(timespec="seconds")

        report_markdown = self._render_markdown(
            title=report_title,
            generated_at=generated_at,
            summary=summary,
            datasets=dataset_reports,
            execution_history=execution_history,
            limitations=limitations,
        )

        return {
            "agent": "ReportAgent",
            "status": status,
            "title": report_title,
            "generated_at": generated_at,
            "dataset": dataset_name,
            "summary": summary,
            "dataset_count": len(dataset_reports),
            "finding_count": len(all_findings),
            "evidence_linked_finding_count": evidence_count,
            "model_linked_finding_count": model_count,
            "datasets": dataset_reports,
            "findings": all_findings,
            "execution_history": execution_history,
            "limitations": limitations,
            "report_markdown": report_markdown,
        }

    @staticmethod
    def _resolve_graph(knowledge_graph):
        """Accept either the project wrapper or a NetworkX graph."""

        if (
            hasattr(knowledge_graph, "nodes")
            and hasattr(knowledge_graph, "out_edges")
        ):
            return knowledge_graph

        graph = getattr(knowledge_graph, "graph", None)

        if (
            graph is not None
            and hasattr(graph, "nodes")
            and hasattr(graph, "out_edges")
        ):
            return graph

        raise TypeError(
            "knowledge_graph must be a KnowledgeGraph instance "
            "or a compatible NetworkX graph."
        )

    @staticmethod
    def _select_datasets(
        node_data,
        relations,
        dataset_name,
    ):
        dataset_nodes = [
            (node_id, attributes)
            for node_id, attributes in node_data.items()
            if attributes.get("type") == "dataset"
        ]

        if dataset_name is not None:
            requested = str(dataset_name).strip().casefold()

            matches = [
                (node_id, attributes)
                for node_id, attributes in dataset_nodes
                if str(
                    attributes.get("name", node_id)
                ).strip().casefold() == requested
                or str(node_id).casefold()
                == f"dataset_{requested}"
            ]

            if matches:
                return [
                    (
                        node_id,
                        str(attributes.get("name", node_id)),
                    )
                    for node_id, attributes in matches
                ]

            # No matching dataset node. Limit the report to findings
            # explicitly tagged with the requested dataset, if any.
            return [(None, str(dataset_name).strip())]

        if dataset_nodes:
            return [
                (
                    node_id,
                    str(attributes.get("name", node_id)),
                )
                for node_id, attributes in dataset_nodes
            ]

        # Older or manually created graphs may have findings without
        # explicit dataset nodes. Allow a fallback to finding metadata.
        finding_datasets = []

        for attributes in node_data.values():
            if attributes.get("type") != "finding":
                continue

            label = str(
                attributes.get("dataset") or "Unknown dataset"
            )

            if label not in finding_datasets:
                finding_datasets.append(label)

        if not finding_datasets:
            return [("unknown_dataset", "Unknown dataset")]

        return [(None, label) for label in finding_datasets]

    @staticmethod
    def _findings_for_dataset(
        node_data,
        relations,
        dataset_id,
        dataset_name,
        all_findings=False,
    ):
        if all_findings:
            return [
                node_id
                for node_id, attributes in node_data.items()
                if attributes.get("type") == "finding"
            ]

        findings = []

        if dataset_id is not None:
            findings.extend(
                relations.get(
                    (dataset_id, "contains_finding"),
                    [],
                )
            )

        # Support graph variants where findings carry dataset metadata
        # but are not linked with contains_finding edges.
        for node_id, attributes in node_data.items():
            if attributes.get("type") != "finding":
                continue

            stored_dataset = attributes.get("dataset")

            if (
                stored_dataset is not None
                and str(stored_dataset).casefold()
                == str(dataset_name).casefold()
            ):
                findings.append(node_id)

        # Preserve ordering while removing duplicates.
        return list(dict.fromkeys(findings))

    def _build_finding_record(
        self,
        finding_id,
        node_data,
        relations,
        dataset_label,
    ):
        attributes = node_data[finding_id]

        evidence_records = []

        for evidence_id in relations.get(
            (finding_id, "supported_by"),
            [],
        ):
            evidence_attributes = node_data.get(
                evidence_id, {}
            )

            if evidence_attributes.get("type") != "evidence":
                continue

            evidence_records.append(
                {
                    "evidence_id": evidence_id,
                    "description": evidence_attributes.get(
                        "description",
                        "No description recorded",
                    ),
                    "method": evidence_attributes.get("method"),
                    "value": self._normalise_value(
                        evidence_attributes.get("value")
                    ),
                }
            )

        model_records = []

        for model_id in relations.get(
            (finding_id, "generated_by"),
            [],
        ):
            model_attributes = node_data.get(model_id, {})

            if model_attributes.get("type") != "model":
                continue

            model_records.append(
                {
                    "model_id": model_id,
                    "model_name": model_attributes.get(
                        "model_name",
                        "Unnamed model",
                    ),
                    "metrics": self._normalise_value(
                        model_attributes.get("metrics", {})
                    ),
                    "accuracy": self._normalise_value(
                        model_attributes.get("accuracy")
                    ),
                }
            )

        confidence = self._normalise_value(
            attributes.get("confidence")
        )

        return {
            "finding_id": finding_id,
            "dataset": dataset_label,
            "description": attributes.get(
                "description",
                "No description recorded",
            ),
            "agent": attributes.get("agent"),
            "task": attributes.get("task"),
            "confidence": confidence,
            "evidence": evidence_records,
            "models": model_records,
        }

    @staticmethod
    def _get_execution_history(
        node_data,
        relations,
        finding_ids,
    ):
        records = []

        if not finding_ids:
            return {
                "count": 0,
                "successful_count": 0,
                "unsuccessful_count": 0,
                "records": [],
            }

        for execution_id, attributes in node_data.items():
            if attributes.get("type") != "execution":
                continue

            produced = set(
                relations.get(
                    (execution_id, "produced"),
                    [],
                )
            )

            if not produced.intersection(finding_ids):
                continue

            records.append(
                {
                    "execution_id": execution_id,
                    "agent": attributes.get("agent"),
                    "task": attributes.get("task"),
                    "dataset": attributes.get("dataset"),
                    "successful": attributes.get("successful"),
                    "confidence": ReportAgent._normalise_value(
                        attributes.get("confidence")
                    ),
                    "model_performance":
                        ReportAgent._normalise_value(
                            attributes.get("model_performance")
                        ),
                    "verification_passed":
                        attributes.get("verification_passed"),
                }
            )

        successful_count = sum(
            record["successful"] is True
            for record in records
        )

        unsuccessful_count = sum(
            record["successful"] is False
            for record in records
        )

        return {
            "count": len(records),
            "successful_count": successful_count,
            "unsuccessful_count": unsuccessful_count,
            "records": records,
        }

    @staticmethod
    def _normalise_value(value):
        """Convert common values to a JSON-friendly representation."""

        if value is None or isinstance(
            value, (str, bool, int)
        ):
            return value

        if isinstance(value, float):
            if value != value or value in (
                float("inf"),
                float("-inf"),
            ):
                return str(value)
            return value

        if isinstance(value, dict):
            return {
                str(key): ReportAgent._normalise_value(item)
                for key, item in value.items()
            }

        if isinstance(value, (list, tuple, set)):
            return [
                ReportAgent._normalise_value(item)
                for item in value
            ]

        return str(value)

    @staticmethod
    def _cell(value):
        """Format one safe Markdown table cell."""

        if value is None:
            return "—"

        if isinstance(value, (dict, list)):
            value = json.dumps(
                value,
                ensure_ascii=False,
                default=str,
            )

        return (
            str(value)
            .replace("|", "\\|")
            .replace("\r", " ")
            .replace("\n", " ")
        )

    def _render_markdown(
        self,
        title,
        generated_at,
        summary,
        datasets,
        execution_history,
        limitations,
    ):
        """Render the structured report as Markdown."""

        lines = [
            f"# {title}",
            "",
            f"**Generated (UTC):** {generated_at}",
            "",
            "## Executive summary",
            "",
            summary,
            "",
        ]

        for dataset in datasets:
            lines.extend(
                [
                    f"## Dataset: {dataset['dataset']}",
                    "",
                    (
                        f"Recorded findings: "
                        f"{dataset['finding_count']}"
                    ),
                    "",
                ]
            )

            if not dataset["findings"]:
                lines.extend(
                    [
                        "No findings are recorded for this dataset.",
                        "",
                    ]
                )
                continue

            for finding in dataset["findings"]:
                lines.extend(
                    [
                        (
                            f"### {finding['finding_id']}: "
                            f"{self._cell(finding['description'])}"
                        ),
                        "",
                        f"- **Source agent:** "
                        f"{self._cell(finding['agent'])}",
                        f"- **Task:** {self._cell(finding['task'])}",
                        f"- **Recorded confidence:** "
                        f"{self._cell(finding['confidence'])}",
                        "",
                    ]
                )

                if finding["evidence"]:
                    lines.extend(
                        [
                            "#### Linked evidence",
                            "",
                            "| Evidence | Method | Recorded value |",
                            "|---|---|---|",
                        ]
                    )

                    for evidence in finding["evidence"]:
                        lines.append(
                            "| "
                            + " | ".join(
                                [
                                    self._cell(
                                        evidence["description"]
                                    ),
                                    self._cell(evidence["method"]),
                                    self._cell(evidence["value"]),
                                ]
                            )
                            + " |"
                        )

                    lines.append("")
                else:
                    lines.extend(
                        [
                            (
                                "_No evidence node is linked "
                                "to this finding in the graph._"
                            ),
                            "",
                        ]
                    )

                if finding["models"]:
                    lines.extend(
                        [
                            "#### Linked model metadata",
                            "",
                            "| Model | Accuracy | Metrics |",
                            "|---|---:|---|",
                        ]
                    )

                    for model in finding["models"]:
                        lines.append(
                            "| "
                            + " | ".join(
                                [
                                    self._cell(model["model_name"]),
                                    self._cell(model["accuracy"]),
                                    self._cell(model["metrics"]),
                                ]
                            )
                            + " |"
                        )

                    lines.append("")

        lines.extend(
            [
                "## Execution history",
                "",
                (
                    f"Recorded executions linked to these findings: "
                    f"{execution_history['count']}."
                ),
                (
                    f"Marked successful: "
                    f"{execution_history['successful_count']}."
                ),
                (
                    f"Marked unsuccessful: "
                    f"{execution_history['unsuccessful_count']}."
                ),
                "",
                "## Interpretation and limitations",
                "",
            ]
        )

        for limitation in limitations:
            lines.append(f"- {limitation}")

        lines.append("")

        return "\n".join(lines)
