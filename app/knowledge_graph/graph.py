import copy
import inspect
import json
import os
import tempfile
import threading
from pathlib import Path

import networkx as nx
from filelock import FileLock
from networkx.readwrite import json_graph


GRAPH_LOCK = threading.RLock()
_SAVE_LOCK = threading.RLock()
_MISSING = object()


def _node_link_kwargs(function):
    parameters = inspect.signature(function).parameters

    if "edges" in parameters:
        return {"edges": "edges"}

    if "link" in parameters:
        return {"link": "edges"}

    return {}


def _read_graph(path):
    """Read a graph file, or return an empty graph if it doesn't exist."""
    if not path.exists():
        return nx.MultiDiGraph()

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    return json_graph.node_link_graph(
        data,
        directed=True,
        multigraph=True,
        **_node_link_kwargs(json_graph.node_link_graph),
    )


def _merge_attribute_changes(
    entity_label,
    base_attributes,
    local_attributes,
    latest_attributes,
):
    """Apply local attribute changes without overwriting concurrent changes."""
    for attribute in set(base_attributes) | set(local_attributes):
        old_value = base_attributes.get(attribute, _MISSING)
        new_value = local_attributes.get(attribute, _MISSING)

        if old_value == new_value:
            continue

        current_value = latest_attributes.get(attribute, _MISSING)

        # The other writer either left this attribute unchanged or
        # independently wrote the same value.
        if (
            current_value != old_value
            and current_value != new_value
        ):
            raise RuntimeError(
                f"Concurrent update conflict for {entity_label}, "
                f"attribute {attribute!r}."
            )

        if new_value is _MISSING:
            latest_attributes.pop(attribute, None)
        else:
            latest_attributes[attribute] = copy.deepcopy(new_value)


def _edge_map(graph):
    """Map each directed multigraph edge identity to its attributes."""
    return {
        (source, target, key): copy.deepcopy(attributes)
        for source, target, key, attributes
        in graph.edges(keys=True, data=True)
    }


def _merge_graphs(base_graph, local_graph, latest_graph):
    """
    Merge local changes into the latest saved graph.

    The baseline is the graph this instance last loaded or saved.
    Changes that conflict with a concurrent writer raise RuntimeError
    instead of silently overwriting the other writer's information.
    """
    merged = copy.deepcopy(latest_graph)

    # Apply modifications and removals of existing nodes.
    for node_id, base_data in base_graph.nodes(data=True):
        base_attributes = copy.deepcopy(dict(base_data))

        if node_id not in local_graph:
            if node_id in merged:
                if dict(merged.nodes[node_id]) != base_attributes:
                    raise RuntimeError(
                        f"Concurrent update conflict while removing "
                        f"node {node_id!r}."
                    )
                merged.remove_node(node_id)
            continue

        local_attributes = copy.deepcopy(
            dict(local_graph.nodes[node_id])
        )

        if local_attributes == base_attributes:
            continue

        if node_id not in merged:
            raise RuntimeError(
                f"Node {node_id!r} was removed by another writer."
            )

        _merge_attribute_changes(
            f"node {node_id!r}",
            base_attributes,
            local_attributes,
            merged.nodes[node_id],
        )

    # Add nodes created locally since the baseline.
    for node_id, local_data in local_graph.nodes(data=True):
        if node_id in base_graph:
            continue

        local_attributes = copy.deepcopy(dict(local_data))

        if node_id not in merged:
            merged.add_node(node_id, **local_attributes)
        elif dict(merged.nodes[node_id]) != local_attributes:
            raise RuntimeError(
                f"Concurrent node ID collision for {node_id!r}; "
                "reload the graph and retry with fresh IDs."
            )

    base_edges = _edge_map(base_graph)
    local_edges = _edge_map(local_graph)
    latest_edges = _edge_map(merged)

    # Apply modifications and removals of existing edges.
    for identity, base_attributes in base_edges.items():
        if identity not in local_edges:
            if identity in latest_edges:
                if latest_edges[identity] != base_attributes:
                    raise RuntimeError(
                        f"Concurrent update conflict while removing "
                        f"edge {identity!r}."
                    )
                merged.remove_edge(*identity)
            continue

        local_attributes = local_edges[identity]

        if local_attributes == base_attributes:
            continue

        if identity not in latest_edges:
            raise RuntimeError(
                f"Edge {identity!r} was removed by another writer."
            )

        source, target, key = identity
        _merge_attribute_changes(
            f"edge {identity!r}",
            base_attributes,
            local_attributes,
            merged[source][target][key],
        )

    # Add locally created edges without overwriting existing edge data.
    latest_edges = _edge_map(merged)

    for identity, local_attributes in local_edges.items():
        if identity in base_edges:
            continue

        if identity not in latest_edges:
            source, target, key = identity
            merged.add_edge(
                source,
                target,
                key=key,
                **copy.deepcopy(local_attributes),
            )
        elif latest_edges[identity] != local_attributes:
            raise RuntimeError(
                f"Concurrent edge ID collision for {identity!r}."
            )

    return merged


class KnowledgeGraph:

    def __init__(self, storage_path=None):
        self.graph = nx.MultiDiGraph()
        self.storage_path = (
            Path(storage_path)
            if storage_path is not None
            else None
        )
        self._baseline_path = self.storage_path
        self._baseline_graph = nx.MultiDiGraph()

        if self.storage_path and self.storage_path.exists():
            self.load()

    def add_node(self, node_id, node_type, **attributes):
        with GRAPH_LOCK:
            self.graph.add_node(
                node_id,
                type=node_type,
                **attributes,
            )

    def add_node_object(self, node):
        with GRAPH_LOCK:
            self.graph.add_node(
                node.node_id,
                type=node.node_type,
                **node.attributes,
            )

    def add_relationship(self, source, relation, target):
        with GRAPH_LOCK:
            self.graph.add_edge(
                source,
                target,
                key=relation,
                relation=relation,
            )

    def get_node(self, node_id):
        """Return independent node attributes."""
        with GRAPH_LOCK:
            return copy.deepcopy(self.graph.nodes[node_id])

    def get_relationships(self, node_id):
        """Return an independent copy of outgoing relationships."""
        with GRAPH_LOCK:
            relationships = list(
                self.graph.out_edges(
                    node_id,
                    keys=True,
                    data=True,
                )
            )
            return copy.deepcopy(relationships)

    def get_graph(self):
        """Return an isolated graph snapshot with frozen topology."""
        with GRAPH_LOCK:
            snapshot = copy.deepcopy(self.graph)
            nx.freeze(snapshot)
            return snapshot

    def save(self, path=None):
        """
        Atomically save a graph with cross-process synchronization.

        Merge non-conflicting changes from this instance into the latest
        file so independent writers don't overwrite one another's nodes.
        """
        destination = (
            Path(path)
            if path is not None
            else self.storage_path
        )

        if destination is None:
            raise ValueError("A storage path must be provided.")

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # The thread lock coordinates threads in this process.
        # FileLock coordinates separate processes using this path.
        with _SAVE_LOCK:
            with FileLock(str(destination) + ".lock"):
                with GRAPH_LOCK:
                    local_graph = copy.deepcopy(self.graph)

                    if self._baseline_path == destination:
                        base_graph = copy.deepcopy(
                            self._baseline_graph
                        )
                    else:
                        base_graph = _read_graph(destination)

                latest_graph = _read_graph(destination)

                merged_graph = _merge_graphs(
                    base_graph,
                    local_graph,
                    latest_graph,
                )

                data = json_graph.node_link_data(
                    merged_graph,
                    **_node_link_kwargs(json_graph.node_link_data),
                )

                temporary_path = None

                try:
                    with tempfile.NamedTemporaryFile(
                        mode="w",
                        encoding="utf-8",
                        dir=destination.parent,
                        prefix=f".{destination.name}.",
                        suffix=".tmp",
                        delete=False,
                    ) as temporary_file:
                        temporary_path = Path(temporary_file.name)

                        json.dump(data, temporary_file, indent=2)
                        temporary_file.flush()
                        os.fsync(temporary_file.fileno())

                    os.replace(temporary_path, destination)
                    temporary_path = None

                finally:
                    if temporary_path is not None:
                        try:
                            temporary_path.unlink(missing_ok=True)
                        except OSError:
                            pass

                with GRAPH_LOCK:
                    self.graph = merged_graph
                    self._baseline_graph = copy.deepcopy(merged_graph)
                    self._baseline_path = destination

    def load(self, path=None):
        """Load graph data and establish a fresh merge baseline."""
        source = (
            Path(path)
            if path is not None
            else self.storage_path
        )

        if source is None:
            raise ValueError("A storage path must be provided.")

        if not source.exists():
            return

        with FileLock(str(source) + ".lock"):
            with GRAPH_LOCK:
                loaded_graph = _read_graph(source)

                self.graph = loaded_graph
                self._baseline_graph = copy.deepcopy(loaded_graph)
                self._baseline_path = source