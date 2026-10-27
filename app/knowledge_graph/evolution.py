import hashlib
import json
import math

from app.knowledge_graph.graph import GRAPH_LOCK


class SelfEvolutionEngine:

    def __init__(self, knowledge_graph):
        self.kg = knowledge_graph

    def evaluate_execution(self, result, verification):
        checks = verification.get("checks", {})
        # Use structural verification when available.
        # Fall back to the legacy field for older callers.
        verification_passed = verification.get(
            "structural_verified",
            verification.get("verified", False),
        )

        confidence = result.confidence

        # Validate confidence before using it.
        valid_confidence = (
            isinstance(confidence, (int, float))
            and not isinstance(confidence, bool)
            and math.isfinite(confidence)
            and 0.0 <= confidence <= 1.0
        )

        model_performance = None
        if result.model:
            model_performance = result.model.metrics.get("accuracy")

        valid_model_performance = (
            model_performance is None
            or (
                isinstance(model_performance, (int, float))
                and not isinstance(model_performance, bool)
                and math.isfinite(model_performance)
                and 0.0 <= model_performance <= 1.0
            )
        )

        successful = (
            verification_passed is True
            and valid_confidence
            and confidence >= 0.7
            and valid_model_performance
        )

        if model_performance is not None and valid_model_performance:
            successful = successful and model_performance >= 0.7

        return {
            "successful": bool(successful),
            "confidence": confidence,
            "model_performance": model_performance,
            "verification_passed": verification_passed,
            "checks": checks,
        }

    def update_knowledge(
        self, result, evaluation, finding_ids, run_id=None
    ):
        """
        Store execution history.

        When run_id is supplied, an identical retry reuses the
        existing execution instead of creating another one.
        """
        with GRAPH_LOCK:
            execution_metadata = {}

            if run_id is not None:
                if not isinstance(run_id, str) or not run_id.strip():
                    raise ValueError(
                        "run_id must be a non-empty string."
                    )

                run_id = run_id.strip()

                payload = json.dumps(
                    {
                        "result": result.model_dump(mode="json"),
                        "evaluation": evaluation,
                        "finding_ids": list(finding_ids),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                )

                execution_fingerprint = hashlib.sha256(
                    payload.encode("utf-8")
                ).hexdigest()

                # Check whether this execution has already been recorded.
                for node_id, data in self.kg.graph.nodes(data=True):
                    if (
                        data.get("type") == "execution"
                        and data.get("run_id") == run_id
                    ):
                        if (
                            data.get("execution_fingerprint")
                            != execution_fingerprint
                        ):
                            raise ValueError(
                                "run_id has already been used for a "
                                "different execution."
                            )

                        return node_id

                execution_metadata = {
                    "run_id": run_id,
                    "execution_fingerprint": execution_fingerprint,
                }

            # Allocate an unused execution ID.
            execution_number = 1

            while (
                f"execution_{execution_number}"
                in self.kg.graph.nodes
            ):
                execution_number += 1

            execution_id = f"execution_{execution_number}"
            
            strategies = sorted({
                evidence.method.strip()
                for evidence in result.evidence
                if isinstance(evidence.method, str)
                and evidence.method.strip()
            })
            
            strategy_reasons = sorted({
                evidence.strategy_reason.strip()
                for evidence in result.evidence
                if isinstance(evidence.strategy_reason, str)
                and evidence.strategy_reason.strip()
            })

            # EDA correlation tasks normally attach one profile. Preserve it
            # on the execution node so future recommendations can retrieve
            # similar analytical contexts without parsing display text.
            strategy_contexts = [
                evidence.strategy_context
                for evidence in result.evidence
                if isinstance(evidence.strategy_context, dict)
                and evidence.strategy_context
            ]
            strategy_context = (
                strategy_contexts[0]
                if len(strategy_contexts) == 1
                else None
            )
            
            self.kg.add_node(
                execution_id,
                "execution",
                agent=result.agent,
                task=result.task,
                dataset=result.dataset,
                successful=evaluation["successful"],
                confidence=evaluation["confidence"],
                model_performance=evaluation["model_performance"],
                verification_passed=evaluation["verification_passed"],
                **execution_metadata,
                strategies=strategies,
                strategy_reasons=strategy_reasons,
                strategy_context=strategy_context,
            )

            # Link the execution only to its own valid findings.
            for finding_id in finding_ids:
                if (
                    finding_id in self.kg.graph.nodes
                    and self.kg.graph.nodes[finding_id].get("type")
                    == "finding"
                ):
                    self.kg.add_relationship(
                        execution_id,
                        "produced",
                        finding_id,
                    )

            return execution_id

    def _get_executions(
        self, successful=None, task=None, dataset=None
    ):
        with GRAPH_LOCK:
            executions = []

            for node_id, data in self.kg.graph.nodes(data=True):
                if data.get("type") != "execution":
                    continue

                if task is not None and data.get("task") != task:
                    continue

                if dataset is not None and data.get("dataset") != dataset:
                    continue

                if (
                    successful is not None
                    and data.get("successful") is not successful
                ):
                    continue

                executions.append({
                    "execution_id": node_id,
                    "agent": data.get("agent"),
                    "task": data.get("task"),
                    "dataset": data.get("dataset"),
                    "strategies": data.get("strategies", []),
                    "strategy_reasons": data.get("strategy_reasons", []),
                    "strategy_context": data.get("strategy_context"),
                    "successful": data.get("successful"),
                    "confidence": data.get("confidence"),
                    "model_performance": data.get("model_performance"),
                })

            return executions

    def get_successful_executions(self):
        return self._get_executions(successful=True)

    def get_failed_executions(self):
        return self._get_executions(successful=False)

    def find_similar_executions(self, task, dataset=None):
        return self._get_executions(
            task=task,
            dataset=dataset,
        )

    
    def recommend_strategy(
        self,
        task,
        dataset=None,
        context=None,
        top_k=5,
        min_similarity=0.60,
    ):
        """Recommend a strategy from successful execution history.

        Without `context`, retain the original task-and-dataset majority
        recommender for compatibility. When context is supplied, retrieve
        successful experiences with similar versioned data profiles and
        weight their strategy votes by contextual similarity.

        This is a retrieval heuristic, not proof that a strategy is
        statistically superior.
        """
        if context is not None:
            return self._recommend_contextual_strategy(
                task=task,
                dataset=dataset,
                context=context,
                top_k=top_k,
                min_similarity=min_similarity,
            )

        executions = self.find_similar_executions(task)

        # Never mix histories from different datasets.
        if dataset is None:
            datasets = {
                execution["dataset"] for execution in executions
            }
            if len(datasets) > 1:
                return {
                    "recommended_strategy": None,
                    "task": task,
                    "dataset": None,
                    "reason": (
                        "Multiple datasets have history. Specify a dataset "
                        "to avoid mixing execution histories."
                    ),
                }
            if len(datasets) == 1:
                dataset = next(iter(datasets))
        else:
            executions = [
                execution for execution in executions
                if execution["dataset"] == dataset
            ]

        # Only use successful executions with recorded strategies.
        strategy_counts = {}
        for execution in executions:
            if execution["successful"] is not True:
                continue

            for strategy in execution.get("strategies", []):
                if isinstance(strategy, str) and strategy.strip():
                    strategy = strategy.strip()
                    strategy_counts[strategy] = (
                        strategy_counts.get(strategy, 0) + 1
                    )

        if not strategy_counts:
            return {
                "recommended_strategy": None,
                "task": task,
                "dataset": dataset,
                "reason": (
                    "No successful execution with a recorded strategy "
                    "exists for this task and dataset."
                ),
            }

        highest_count = max(strategy_counts.values())
        leaders = sorted(
            strategy for strategy, count in strategy_counts.items()
            if count == highest_count
        )

        # A tie is inconclusive; let the data-based selector decide.
        if len(leaders) > 1:
            return {
                "recommended_strategy": None,
                "task": task,
                "dataset": dataset,
                "strategy_counts": strategy_counts,
                "reason": (
                    "Historical strategies are tied. Use the data-based "
                    "selector rather than making an arbitrary choice."
                ),
            }

        strategy = leaders[0]
        return {
            "recommended_strategy": strategy,
            "task": task,
            "dataset": dataset,
            "strategy_counts": strategy_counts,
            "reason": (
                f"{strategy} appears most often in successful recorded "
                "executions for this task and dataset. This is historical "
                "precedent, not proof of statistical superiority."
            ),
        }

    @staticmethod
    def _context_features(context):
        """Return validated numeric features from a supported context."""
        if not isinstance(context, dict) or context.get("version") != 1:
            return None

        features = context.get("features")
        if not isinstance(features, dict):
            return None

        allowed = {
            "log10_sample_size": (0.0, 2.5),
            "missing_fraction": (0.0, 1.0),
            "max_abs_skew": (0.0, 4.0),
            "mean_abs_skew": (0.0, 4.0),
            "max_iqr_outlier_fraction": (0.0, 1.0),
            "pearson_spearman_gap": (0.0, 2.0),
            "correlation_gap_available": (0.0, 1.0),
            "constant_column_fraction": (0.0, 1.0),
        }
        validated = {}
        for name, (minimum, scale) in allowed.items():
            value = features.get(name)
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
            ):
                validated[name] = min(
                    max(float(value), minimum),
                    minimum + scale,
                )

        return validated or None

    @classmethod
    def _context_similarity(cls, left, right):
        """Return [0, 1] similarity for version-1 analytical profiles."""
        left_features = cls._context_features(left)
        right_features = cls._context_features(right)
        if left_features is None or right_features is None:
            return None

        # Scales approximate meaningful changes for each profile feature.
        # We cap each normalized distance at 1 so no one feature dominates.
        scales = {
            "log10_sample_size": 2.5,
            "missing_fraction": 1.0,
            "max_abs_skew": 4.0,
            "mean_abs_skew": 4.0,
            "max_iqr_outlier_fraction": 1.0,
            "pearson_spearman_gap": 2.0,
            "correlation_gap_available": 1.0,
            "constant_column_fraction": 1.0,
        }
        weights = {
            "log10_sample_size": 0.75,
            "missing_fraction": 1.0,
            "max_abs_skew": 1.5,
            "mean_abs_skew": 1.0,
            "max_iqr_outlier_fraction": 1.25,
            "pearson_spearman_gap": 1.25,
            "correlation_gap_available": 0.25,
            "constant_column_fraction": 0.75,
        }

        shared = set(left_features) & set(right_features)
        if not shared:
            return None

        weighted_distance = 0.0
        weight_total = 0.0
        for name in shared:
            weight = weights[name]
            normalized_difference = min(
                abs(left_features[name] - right_features[name])
                / scales[name],
                1.0,
            )
            weighted_distance += weight * normalized_difference
            weight_total += weight

        if weight_total == 0:
            return None
        return max(0.0, min(1.0, 1.0 - weighted_distance / weight_total))

    def _recommend_contextual_strategy(
        self, task, dataset, context, top_k, min_similarity
    ):
        """Retrieve similar historical contexts and weight strategy votes.

        Repeated executions of an identical context in one dataset are
        grouped into a single experience. This prevents repeatedly clicking
        Run from manufacturing extra votes for the same decision context.
        Different strategies observed for one context share that context's
        vote equally instead of letting repetitions dominate.
        """
        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, int)
            or top_k < 1
        ):
            raise ValueError("top_k must be a positive integer.")
        if (
            isinstance(min_similarity, bool)
            or not isinstance(min_similarity, (int, float))
            or not math.isfinite(float(min_similarity))
            or not 0.0 <= float(min_similarity) <= 1.0
        ):
            raise ValueError("min_similarity must be between 0 and 1.")

        current_features = self._context_features(context)
        if current_features is None:
            return {
                "recommended_strategy": None,
                "task": task,
                "dataset": dataset,
                "retrieval_mode": "contextual",
                "reason": (
                    "The current analytical context is missing supported "
                    "version-1 numeric profile features. Use the data-based "
                    "selector instead."
                ),
            }

        executions = []
        # Do not restrict context retrieval to the same dataset name. Similar
        # verified task contexts may be useful across datasets; source dataset
        # and profile details remain available for auditing.
        with GRAPH_LOCK:
            for node_id, data in self.kg.graph.nodes(data=True):
                if data.get("type") != "execution":
                    continue
                if data.get("task") != task:
                    continue
                if data.get("successful") is not True:
                    continue
                strategies = data.get("strategies", [])
                if not isinstance(strategies, list):
                    continue
                strategies = sorted({
                    item.strip()
                    for item in strategies
                    if isinstance(item, str) and item.strip()
                })
                if not strategies:
                    continue

                historical_context = data.get("strategy_context")
                similarity = self._context_similarity(
                    context, historical_context
                )
                if (
                    similarity is None
                    or similarity < float(min_similarity)
                ):
                    continue

                features = self._context_features(historical_context) or {}
                columns = historical_context.get("columns", [])
                if not isinstance(columns, list):
                    columns = []
                columns = tuple(sorted(
                    str(column) for column in columns
                ))
                # A rounded profile key collapses repeated executions of
                # the same dataset/column-pair/context, but preserves
                # materially changed contexts as separate experiences.
                profile_key = tuple(
                    (name, round(value, 3))
                    for name, value in sorted(features.items())
                )
                key = (data.get("dataset"), columns, profile_key)
                executions.append({
                    "key": key,
                    "execution_id": node_id,
                    "dataset": data.get("dataset"),
                    "columns": list(columns),
                    "context": historical_context,
                    "similarity": float(similarity),
                    "strategies": strategies,
                })

        if not executions:
            return {
                "recommended_strategy": None,
                "task": task,
                "dataset": dataset,
                "retrieval_mode": "contextual",
                "contextual_matches": 0,
                "reason": (
                    "No successful historical experiences met the context "
                    "similarity threshold. Use the data-based selector."
                ),
            }

        # Group duplicate executions into unique historical experiences.
        grouped = {}
        for item in executions:
            group = grouped.setdefault(
                item["key"],
                {
                    "key": item["key"],
                    "dataset": item["dataset"],
                    "columns": item["columns"],
                    "context": item["context"],
                    "similarity": item["similarity"],
                    "strategies": set(),
                    "execution_ids": [],
                },
            )
            group["similarity"] = max(
                group["similarity"], item["similarity"]
            )
            group["strategies"].update(item["strategies"])
            group["execution_ids"].append(item["execution_id"])

        def _execution_number(value):
            try:
                return int(value.rsplit("_", 1)[-1])
            except (AttributeError, ValueError):
                return -1

        experiences = list(grouped.values())
        experiences.sort(
            key=lambda item: (
                item["similarity"],
                item["dataset"] == dataset,
                max(_execution_number(x) for x in item["execution_ids"]),
            ),
            reverse=True,
        )
        nearest = experiences[:top_k]

        weighted_votes = {}
        for experience in nearest:
            strategies = sorted(experience["strategies"])
            if not strategies:
                continue
            weight = experience["similarity"] ** 2
            per_strategy_weight = weight / len(strategies)
            for strategy in strategies:
                weighted_votes[strategy] = (
                    weighted_votes.get(strategy, 0.0)
                    + per_strategy_weight
                )

        if not weighted_votes:
            return {
                "recommended_strategy": None,
                "task": task,
                "dataset": dataset,
                "retrieval_mode": "contextual",
                "contextual_matches": len(nearest),
                "reason": "Similar contexts did not contain usable strategies.",
            }

        total_weight = sum(weighted_votes.values())
        ordered_votes = sorted(
            weighted_votes.items(),
            key=lambda item: (-item[1], item[0]),
        )
        best_strategy, best_weight = ordered_votes[0]
        support = best_weight / total_weight
        tied = (
            len(ordered_votes) > 1
            and math.isclose(
                ordered_votes[0][1], ordered_votes[1][1],
                rel_tol=1e-9, abs_tol=1e-9,
            )
        )

        nearest_similarity = nearest[0]["similarity"]
        mean_similarity = sum(
            item["similarity"] for item in nearest
        ) / len(nearest)
        common = {
            "task": task,
            "dataset": dataset,
            "retrieval_mode": "contextual",
            "contextual_matches": len(nearest),
            "mean_similarity": round(mean_similarity, 3),
            "nearest_similarity": round(nearest_similarity, 3),
            "strategy_support": {
                strategy: round(weight / total_weight, 3)
                for strategy, weight in ordered_votes
            },
            "supporting_experiences": [
                {
                    "dataset": item["dataset"],
                    "columns": item["columns"],
                    "similarity": round(item["similarity"], 3),
                    "strategies": sorted(item["strategies"]),
                }
                for item in nearest
            ],
        }

        if tied or support < 0.55:
            return {
                **common,
                "recommended_strategy": None,
                "reason": (
                    "Contextually similar experiences do not provide a "
                    "clear strategy preference. Use the data-based selector."
                ),
            }

        return {
            **common,
            "recommended_strategy": best_strategy,
            "reason": (
                f"Contextual memory selected {best_strategy} from "
                f"{len(nearest)} distinct similar experience(s); nearest "
                f"profile similarity={nearest_similarity:.2f}, weighted "
                f"support={support:.0%}. Similarity-based historical "
                "precedent is advisory, not proof of statistical superiority."
            ),
        }


    def _recommend(self, task, dataset=None):
        executions = self.find_similar_executions(task)

        # Keep recommendations isolated to the requested dataset.
        if dataset is not None:
            executions = [
                execution
                for execution in executions
                if execution["dataset"] == dataset
            ]
        else:
            datasets = {
                execution["dataset"]
                for execution in executions
            }

            if len(datasets) > 1:
                return {
                    "recommended_agent": None,
                    "task": task,
                    "dataset": None,
                    "reason": (
                        "Multiple datasets have execution history for this "
                        "task. Specify a dataset to avoid mixing histories."
                    ),
                }

            if len(datasets) == 1:
                dataset = next(iter(datasets))

        if not executions:
            return {
                "recommended_agent": None,
                "task": task,
                "dataset": dataset,
                "reason": (
                    "No previous execution found for this task and dataset."
                ),
            }

        # Execution IDs are allocated sequentially by update_knowledge.
        # Sort numerically so the latest execution for each agent wins.
        def execution_order(execution):
            node_id = execution["execution_id"]
            prefix = "execution_"

            if not node_id.startswith(prefix):
                return -1

            suffix = node_id[len(prefix):]
            return int(suffix) if suffix.isdigit() else -1

        executions = sorted(executions, key=execution_order)

        # Keep only each agent's latest outcome for this task and dataset.
        latest_by_agent = {}

        for execution in executions:
            agent = execution["agent"]

            if agent is not None:
                latest_by_agent[agent] = execution

        latest_executions = list(latest_by_agent.values())

        # An agent is currently failed only when its latest execution
        # was not successful. Earlier failures do not block recovery.
        failed_agents = {
            agent
            for agent, execution in latest_by_agent.items()
            if execution["successful"] is not True
        }

        available = [
            execution
            for execution in latest_executions
            if execution["successful"] is True
            and isinstance(execution["confidence"], (int, float))
            and not isinstance(execution["confidence"], bool)
            and math.isfinite(execution["confidence"])
        ]

        if not available:
            return {
                "recommended_agent": None,
                "task": task,
                "dataset": dataset,
                "reason": (
                    "No eligible successful latest execution is available."
                ),
                "failed_agents": sorted(failed_agents),
            }

                # Compare like with like: never rank model accuracy against
        # confidence, since they represent different measurements.
        candidates_with_model_performance = [
            execution
            for execution in available
            if (
                isinstance(
                    execution["model_performance"],
                    (int, float),
                )
                and not isinstance(
                    execution["model_performance"],
                    bool,
                )
                and math.isfinite(execution["model_performance"])
                and 0.0 <= execution["model_performance"] <= 1.0
            )
        ]

        if candidates_with_model_performance:
            best = max(
                candidates_with_model_performance,
                key=lambda execution: execution["model_performance"],
            )
            ranking_metric = "model performance"
        else:
            best = max(
                available,
                key=lambda execution: execution["confidence"],
            )
            ranking_metric = "confidence"

        return {
            "recommended_agent": best["agent"],
            "task": task,
            "dataset": dataset,
                        "reason": (
                f"Recommended using {ranking_metric} among eligible "
                "latest successful executions for the selected task "
                "and dataset."
            ),
            "confidence": best["confidence"],
            "model_performance": best["model_performance"],
            "failed_agents": sorted(failed_agents),
        }

    def recommend_workflow(self, task, dataset=None):
        return self._recommend(task, dataset)

    def get_recommended_agent(self, task, dataset=None):
        return self.recommend_workflow(
            task, dataset
        ).get("recommended_agent")

    def recommend_agent(self, task, dataset=None):
        return self._recommend(task, dataset)