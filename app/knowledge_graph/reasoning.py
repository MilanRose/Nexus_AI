
from app.knowledge_graph.graph import GRAPH_LOCK


class GraphReasoner:

    def __init__(self, knowledge_graph):
        self.kg = knowledge_graph

    def get_supporting_evidence(self, finding_id):
        with GRAPH_LOCK:
            evidence = []

            for _, target, data in self.kg.graph.out_edges(
                finding_id,
                data=True
            ):
                if data.get("relation") == "supported_by":
                    evidence_data = dict(self.kg.get_node(target))

                    evidence.append({
                        "evidence_id": target,
                        "details": evidence_data
                    })

            return evidence

    def get_related_nodes(self, node_id):
        with GRAPH_LOCK:
            relationships = []

            # Include outgoing relationships while preserving direction.
            for source, target, data in self.kg.graph.out_edges(
                node_id,
                data=True
            ):
                relationships.append({
                    "source": source,
                    "relation": data.get("relation"),
                    "target": target
                })

            # Include incoming relationships while preserving direction.
            for source, target, data in self.kg.graph.in_edges(
                node_id,
                data=True
            ):
                relationships.append({
                    "source": source,
                    "relation": data.get("relation"),
                    "target": target
                })

            return relationships

    def get_finding_context(self, finding_id):
        with GRAPH_LOCK:
            context = {
                "finding": finding_id,
                "features": [],
                "datasets": []
            }

            # Find features supporting the finding.
            for source, _, data in self.kg.graph.in_edges(
                finding_id,
                data=True
            ):
                if data.get("relation") == "supports":
                    if source not in context["features"]:
                        context["features"].append(source)

                    # Find datasets containing those features.
                    for dataset_source, _, dataset_data in (
                        self.kg.graph.in_edges(source, data=True)
                    ):
                        if (
                            dataset_data.get("relation") == "contains"
                            and dataset_source not in context["datasets"]
                        ):
                            context["datasets"].append(dataset_source)

            # Also find datasets directly connected to the finding.
            for source, _, data in self.kg.graph.in_edges(
                finding_id,
                data=True
            ):
                if (
                    data.get("relation") == "contains_finding"
                    and self.kg.graph.nodes[source].get("type") == "dataset"
                    and source not in context["datasets"]
                ):
                    context["datasets"].append(source)

            return context

    def get_provenance_chain(self, finding_id):
        with GRAPH_LOCK:
            finding_data = dict(self.kg.get_node(finding_id))

            chain = {
                "finding": {
                    "id": finding_id,
                    "details": finding_data
                },
                "features": [],
                "datasets": [],
                "evidence": [],
                "models": []
            }

            seen_features = set()
            seen_datasets = set()
            seen_evidence = set()
            seen_models = set()

            # Find incoming relationships to the finding.
            for source, _, data in self.kg.graph.in_edges(
                finding_id,
                data=True
            ):
                relation = data.get("relation")

                # Feature -> Finding
                if relation == "supports":
                    if source not in seen_features:
                        seen_features.add(source)
                        chain["features"].append({
                            "id": source,
                            "details": dict(self.kg.get_node(source))
                        })

                    # Dataset -> Feature
                    for dataset_source, _, dataset_data in (
                        self.kg.graph.in_edges(source, data=True)
                    ):
                        if (
                            dataset_data.get("relation") == "contains"
                            and dataset_source not in seen_datasets
                        ):
                            seen_datasets.add(dataset_source)
                            chain["datasets"].append({
                                "id": dataset_source,
                                "details": dict(
                                    self.kg.get_node(dataset_source)
                                )
                            })

                # Direct Dataset -> Finding provenance
                elif (
                    relation == "contains_finding"
                    and self.kg.graph.nodes[source].get("type") == "dataset"
                    and source not in seen_datasets
                ):
                    seen_datasets.add(source)
                    chain["datasets"].append({
                        "id": source,
                        "details": dict(self.kg.get_node(source))
                    })

            # Find outgoing relationships from the finding.
            for _, target, data in self.kg.graph.out_edges(
                finding_id,
                data=True
            ):
                relation = data.get("relation")

                # Finding -> Evidence
                if (
                    relation == "supported_by"
                    and target not in seen_evidence
                ):
                    seen_evidence.add(target)
                    chain["evidence"].append({
                        "id": target,
                        "details": dict(self.kg.get_node(target))
                    })

                # Finding -> Model
                elif (
                    relation == "generated_by"
                    and target not in seen_models
                ):
                    seen_models.add(target)
                    chain["models"].append({
                        "id": target,
                        "details": dict(self.kg.get_node(target))
                    })

            return chain
