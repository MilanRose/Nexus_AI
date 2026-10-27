from concurrent.futures import ThreadPoolExecutor

import networkx as nx
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import multiprocessing
import threading

import app.knowledge_graph.graph as graph_module

import pytest

from app.knowledge_graph.graph import KnowledgeGraph


def test_graph_persistence():
    with TemporaryDirectory() as temp_dir:
        storage_path = Path(temp_dir) / "knowledge_graph.json"

        # Create a graph with multiple relationships
        kg = KnowledgeGraph(storage_path)
        kg.add_node(
            "dataset_1",
            "dataset",
            name="customer_churn.csv"
        )
        kg.add_node(
            "finding_1",
            "finding",
            confidence=0.84
        )
        kg.add_relationship(
            "dataset_1",
            "contains_finding",
            "finding_1"
        )
        kg.add_relationship(
            "dataset_1",
            "supports",
            "finding_1"
        )

        # Save and verify the file
        kg.save()

        assert storage_path.exists()
        assert storage_path.stat().st_size > 0

        # Load into a fresh graph and verify persistence
        loaded_kg = KnowledgeGraph(storage_path)

        assert "dataset_1" in loaded_kg.graph.nodes
        assert "finding_1" in loaded_kg.graph.nodes
        assert loaded_kg.get_node("dataset_1")["name"] == (
            "customer_churn.csv"
        )
        assert loaded_kg.get_node("finding_1")["confidence"] == 0.84

        relationships = loaded_kg.graph.get_edge_data(
            "dataset_1",
            "finding_1"
        )

        assert relationships is not None
        assert len(relationships) == 2

        relation_types = {
            data["relation"] for data in relationships.values()
        }

        assert relation_types == {"contains_finding", "supports"}

        # Record the original file before attempting a failed save
        original_contents = storage_path.read_bytes()

        # Add a value that JSON cannot serialize
        kg.graph.nodes["finding_1"]["unsupported_value"] = object()

        with pytest.raises(TypeError):
            kg.save()

        # The original file must remain unchanged
        assert storage_path.read_bytes() == original_contents

        # The previous file must still be valid and loadable
        preserved_kg = KnowledgeGraph(storage_path)

        assert "dataset_1" in preserved_kg.graph.nodes
        assert "finding_1" in preserved_kg.graph.nodes
        assert (
            preserved_kg.get_node("finding_1")["confidence"] == 0.84
        )

        # Confirm the file remains valid JSON
        with storage_path.open("r", encoding="utf-8") as file:
            json.load(file)

        print("GRAPH PERSISTENCE AND FAILED-SAVE TEST PASSED")


def test_concurrent_saves_preserve_valid_graph_file():
    with TemporaryDirectory() as temp_dir:
        storage_path = Path(temp_dir) / "concurrent_graph.json"
        kg = KnowledgeGraph(storage_path)

        kg.add_node("dataset_1", "dataset", name="data.csv")

        for index in range(20):
            finding_id = f"finding_{index}"
            kg.add_node(
                finding_id,
                "finding",
                confidence=0.8
            )
            kg.add_relationship(
                "dataset_1",
                "contains_finding",
                finding_id
            )

        # Multiple threads save the same graph concurrently.
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(kg.save)
                for _ in range(10)
            ]

            for future in futures:
                future.result()

        # The resulting file must remain valid JSON and loadable.
        with storage_path.open("r", encoding="utf-8") as file:
            json.load(file)

        loaded_kg = KnowledgeGraph(storage_path)

        assert "dataset_1" in loaded_kg.graph.nodes

        for index in range(20):
            finding_id = f"finding_{index}"
            assert finding_id in loaded_kg.graph.nodes
            assert loaded_kg.graph.has_edge(
                "dataset_1",
                finding_id
            )

def test_get_node_returns_an_independent_copy():
    kg = KnowledgeGraph()

    kg.add_node(
        "finding_1",
        "finding",
        confidence=0.84,
        metadata={"tags": ["original"]},
    )

    node = kg.get_node("finding_1")

    # Changing the returned data must not alter the actual graph.
    node["confidence"] = 0.1
    node["metadata"]["tags"].append("external-change")

    actual_node = kg.get_node("finding_1")

    assert actual_node["confidence"] == 0.84
    assert actual_node["metadata"]["tags"] == ["original"]


def test_get_graph_returns_an_isolated_read_only_snapshot():
    kg = KnowledgeGraph()

    kg.add_node(
        "finding_1",
        "finding",
        metadata={"tags": ["original"]},
    )

    snapshot = kg.get_graph()

    # The snapshot must not allow topology changes.
    with pytest.raises(nx.NetworkXError):
        snapshot.add_node("unexpected_node", type="finding")

    # Its attributes must not mutate the original graph either.
    snapshot.nodes["finding_1"]["metadata"]["tags"].append(
        "external-change"
    )

    assert "unexpected_node" not in kg.graph
    assert kg.get_node("finding_1")["metadata"]["tags"] == ["original"]

def _process_add_node_and_save(storage_path, node_id, barrier):
    """Worker used to test concurrent writes from separate processes."""
    kg = KnowledgeGraph(storage_path)

    # Ensure both workers loaded the same initial graph.
    barrier.wait(timeout=20)

    kg.add_node(
        node_id,
        "worker_result",
        worker=node_id,
    )
    kg.save()


def test_separate_process_saves_preserve_both_workers(tmp_path):
    storage_path = tmp_path / "multiprocess_graph.json"

    initial_graph = KnowledgeGraph(storage_path)
    initial_graph.add_node(
        "seed",
        "dataset",
        name="initial.csv",
    )
    initial_graph.save()

    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2)

    workers = [
        context.Process(
            target=_process_add_node_and_save,
            args=(str(storage_path), f"worker_{index}", barrier),
        )
        for index in range(2)
    ]

    for worker in workers:
        worker.start()

    for worker in workers:
        worker.join(timeout=30)

    try:
        for worker in workers:
            assert not worker.is_alive(), "Worker process timed out."
            assert worker.exitcode == 0, (
                f"Worker process failed with exit code {worker.exitcode}."
            )
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5)

    loaded_graph = KnowledgeGraph(storage_path)

    assert "seed" in loaded_graph.graph
    assert "worker_0" in loaded_graph.graph
    assert "worker_1" in loaded_graph.graph

def test_load_does_not_overwrite_concurrent_node_addition(
    tmp_path,
    monkeypatch,
):
    storage_path = tmp_path / "load_race.json"

    saved_graph = KnowledgeGraph(storage_path)
    saved_graph.add_node("seed", "dataset", name="seed.csv")
    saved_graph.save()

    live_graph = KnowledgeGraph()

    read_finished = threading.Event()
    allow_load_to_finish = threading.Event()
    add_started = threading.Event()
    errors = []

    original_read_graph = graph_module._read_graph

    def paused_read_graph(path):
        loaded = original_read_graph(path)
        read_finished.set()

        if not allow_load_to_finish.wait(timeout=10):
            raise TimeoutError("Test timed out waiting to finish loading.")

        return loaded

    monkeypatch.setattr(
        graph_module,
        "_read_graph",
        paused_read_graph,
    )

    def load_graph():
        try:
            live_graph.load(storage_path)
        except Exception as exc:
            errors.append(exc)

    def add_node():
        add_started.set()
        try:
            live_graph.add_node(
                "concurrent_finding",
                "finding",
                confidence=0.9,
            )
        except Exception as exc:
            errors.append(exc)

    loader = threading.Thread(target=load_graph)
    loader.start()

    assert read_finished.wait(timeout=5)

    adder = threading.Thread(target=add_node)
    adder.start()

    try:
        assert add_started.wait(timeout=5)
    finally:
        allow_load_to_finish.set()

    loader.join(timeout=10)
    adder.join(timeout=10)

    assert not loader.is_alive()
    assert not adder.is_alive()
    assert not errors

    assert "seed" in live_graph.graph
    assert "concurrent_finding" in live_graph.graph