import hashlib
import json
import re

from app.knowledge_graph.graph import GRAPH_LOCK
from app.schemas import AgentResult


class KnowledgeGraphBuilder:

    def __init__(self, knowledge_graph):
        self.kg = knowledge_graph

    def add_agent_result(self, result: AgentResult, run_id=None):
        """Add an agent result, optionally making retries idempotent."""
        with GRAPH_LOCK:
            return self._add_agent_result(result, run_id)

    def _add_agent_result(self, result: AgentResult, run_id=None):
        # Validate evidence references before modifying the graph.
        for evidence_item in result.evidence:
            for finding_index in evidence_item.finding_indices:
                if not 0 <= finding_index < len(result.findings):
                    raise ValueError(
                        f"Evidence '{evidence_item.description}' refers "
                        f"to invalid finding index {finding_index}. "
                        f"Valid indices are 0 to "
                        f"{len(result.findings) - 1}."
                    )

        run_metadata = {}

        # Check whether this result is a retry of an existing run.
        if run_id is not None:
            if not isinstance(run_id, str) or not run_id.strip():
                raise ValueError(
                    "run_id must be a non-empty string."
                )

            run_id = run_id.strip()

            payload = json.dumps(
                result.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )

            result_fingerprint = hashlib.sha256(
                payload.encode("utf-8")
            ).hexdigest()

            existing_run_nodes = [
                (node_id, data)
                for node_id, data in self.kg.graph.nodes(data=True)
                if data.get("type") in {
                    "finding",
                    "model",
                    "evidence",
                }
                and data.get("run_id") == run_id
            ]

            if existing_run_nodes:
                # The same ID cannot represent different result data.
                if any(
                    data.get("result_fingerprint") != result_fingerprint
                    for _, data in existing_run_nodes
                ):
                    raise ValueError(
                        "run_id has already been used for different "
                        "analysis results."
                    )

                existing_findings = sorted(
                    (
                        node_id
                        for node_id, data in existing_run_nodes
                        if data.get("type") == "finding"
                    ),
                    key=lambda node_id: self.kg.graph.nodes[
                        node_id
                    ].get("result_index", 0),
                )

                # Ensure that the original result was fully stored.
                expected_model_count = (
                    len(result.findings)
                    if result.model is not None
                    else 0
                )

                existing_model_count = sum(
                    1
                    for _, data in existing_run_nodes
                    if data.get("type") == "model"
                )

                existing_evidence_count = sum(
                    1
                    for _, data in existing_run_nodes
                    if data.get("type") == "evidence"
                )

                counts_match = (
                    len(existing_findings) == len(result.findings)
                    and existing_model_count == expected_model_count
                    and existing_evidence_count == len(result.evidence)
                )

                if not counts_match:
                    raise RuntimeError(
                        "The existing run contains incomplete graph "
                        "entries. Investigate the graph state before "
                        "retrying with a new run_id."
                    )

                return existing_findings

            run_metadata = {
                "run_id": run_id,
                "result_fingerprint": result_fingerprint,
            }

        # Find a result number that avoids node-ID collisions.
        result_number = 1

        def candidate_node_ids(number):
            node_ids = [
                f"finding_{number}_{i}"
                for i in range(1, len(result.findings) + 1)
            ]

            if result.model is not None:
                node_ids.extend(
                    f"model_{number}_{i}"
                    for i in range(1, len(result.findings) + 1)
                )

            node_ids.extend(
                f"evidence_{number}_{i}"
                for i in range(1, len(result.evidence) + 1)
            )

            return node_ids

        while any(
            node_id in self.kg.graph
            for node_id in candidate_node_ids(result_number)
        ):
            result_number += 1

        created_finding_ids = []

        # Create or reuse the dataset node.
        dataset_id = f"dataset_{result.dataset}"

        self.kg.add_node(
            dataset_id,
            "dataset",
            name=result.dataset,
        )

        # Create finding nodes.
        for finding_index, finding_description in enumerate(
            result.findings
        ):
            finding_id = (
                f"finding_{result_number}_{finding_index + 1}"
            )

            self.kg.add_node(
                finding_id,
                "finding",
                description=finding_description,
                confidence=result.confidence,
                task=result.task,
                agent=result.agent,
                result_index=finding_index,
                **run_metadata,
            )

            created_finding_ids.append(finding_id)

            self.kg.add_relationship(
                dataset_id,
                "contains_finding",
                finding_id,
            )

            # Match complete, non-empty feature names.
            for raw_feature_name in result.features:
                feature_name = raw_feature_name.strip()

                if not feature_name:
                    continue

                if not re.search(
                    rf"(?<!\w){re.escape(feature_name)}(?!\w)",
                    finding_description,
                    flags=re.IGNORECASE,
                ):
                    continue

                legacy_feature_id = (
                    f"feature_{result.dataset}_{feature_name}"
                )

                existing_data = self.kg.graph.nodes.get(
                    legacy_feature_id, {}
                )

                same_feature = (
                    existing_data.get("type") == "feature"
                    and existing_data.get("dataset") == result.dataset
                    and existing_data.get("name") == feature_name
                )

                if (
                    legacy_feature_id not in self.kg.graph
                    or same_feature
                ):
                    feature_id = legacy_feature_id
                else:
                    identity = (
                        f"{result.dataset}\0{feature_name}"
                    )
                    suffix = hashlib.sha256(
                        identity.encode("utf-8")
                    ).hexdigest()[:12]

                    feature_id = f"{legacy_feature_id}_{suffix}"

                    # Avoid collisions with other existing graph nodes.
                    while feature_id in self.kg.graph:
                        existing_data = self.kg.graph.nodes[
                            feature_id
                        ]

                        if (
                            existing_data.get("type") == "feature"
                            and existing_data.get("dataset")
                            == result.dataset
                            and existing_data.get("name") == feature_name
                        ):
                            break

                        suffix = hashlib.sha256(
                            f"{identity}\0{suffix}".encode("utf-8")
                        ).hexdigest()[:12]

                        feature_id = (
                            f"{legacy_feature_id}_{suffix}"
                        )

                self.kg.add_node(
                    feature_id,
                    "feature",
                    name=feature_name,
                    dataset=result.dataset,
                )

                self.kg.add_relationship(
                    dataset_id,
                    "contains",
                    feature_id,
                )

                self.kg.add_relationship(
                    feature_id,
                    "supports",
                    finding_id,
                )

            # Create a model node when a model is available.
            if result.model is not None:
                model_id = (
                    f"model_{result_number}_{finding_index + 1}"
                )

                self.kg.add_node(
                    model_id,
                    "model",
                    model_name=result.model.name,
                    accuracy=result.model.metrics.get("accuracy"),
                    metrics=result.model.metrics,
                    **run_metadata,
                )

                self.kg.add_relationship(
                    finding_id,
                    "generated_by",
                    model_id,
                )

        # Connect evidence only to its specified findings.
        for evidence_index, evidence_item in enumerate(
            result.evidence,
            start=1,
        ):
            evidence_id = (
                f"evidence_{result_number}_{evidence_index}"
            )

            self.kg.add_node(
                evidence_id,
                "evidence",
                description=evidence_item.description,
                method=evidence_item.method,
                value=evidence_item.value,
                strategy_reason=evidence_item.strategy_reason,
                strategy_context=evidence_item.strategy_context,
                **run_metadata,
            )

            for finding_index in evidence_item.finding_indices:
                finding_id = created_finding_ids[finding_index]

                self.kg.add_relationship(
                    finding_id,
                    "supported_by",
                    evidence_id,
                )

        return created_finding_ids