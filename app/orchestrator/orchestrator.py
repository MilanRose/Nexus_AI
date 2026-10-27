
"""Orchestrate the framework's agent workflows."""

from pathlib import Path
from uuid import uuid4

from app.orchestrator.router import AdaptiveRouter
from app.analysis_adapter import analysis_result_to_agent_result
from app.confidence_policy import estimate_confidence
from app.schemas import AgentResult, Evidence
from app.agents.data_agent import DataAgent
from app.agents.cleaning_agent import DataCleaningAgent
from app.agents.explainability_agent import ExplainabilityAgent
from app.agents.ml_agent import MLAgent
from app.agents.report_agent import ReportAgent
from app.knowledge_graph.builder import KnowledgeGraphBuilder
from app.knowledge_graph.evolution import SelfEvolutionEngine
from app.knowledge_graph.graph import KnowledgeGraph
from app.verification.fact_checker import FactChecker
from src.analysis import run_analysis


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Orchestrator:
    """Route requests to available agents and return consistent results."""

    def __init__(
        self,
        dataset_path="data/titanic.csv",
        graph_path=None,
    ):
        self.router = AdaptiveRouter()

        dataset = Path(dataset_path)
        if not dataset.is_absolute():
            dataset = PROJECT_ROOT / dataset

        self.dataset_path = dataset

        if graph_path is None:
            graph_path = (
                PROJECT_ROOT
                / "outputs"
                / "knowledge_graph.json"
            )

        self.knowledge_graph = KnowledgeGraph(
            storage_path=graph_path
        )

        self.graph_builder = KnowledgeGraphBuilder(
            self.knowledge_graph
        )

        self.fact_checker = FactChecker(
            self.knowledge_graph
        )

        self.evolution_engine = SelfEvolutionEngine(
            self.knowledge_graph
        )

        self.data_agent = DataAgent()
        self.cleaning_agent = DataCleaningAgent()
        self.ml_agent = MLAgent()
        self.report_agent = ReportAgent()
        self.explainability_agent = ExplainabilityAgent()

    def _response(
        self,
        user_request,
        selected_agent,
        status,
        *,
        message="",
        dataset=None,
        task_count=0,
        insights=None,
        results=None,
        errors=None,
        run_id=None,
    ):
        """Return a consistent response structure."""

        return {
            "user_request": user_request,
            "selected_agent": selected_agent,
            "status": status,
            "message": message,
            "dataset": dataset,
            "task_count": task_count,
            "insights": insights or [],
            "results": results or [],
            "errors": errors or [],
            "run_id": run_id,
        }

    def _verify_and_record(
        self,
        agent_result,
        run_id,
    ):
        """Store findings, check graph structure, and record execution."""

        finding_ids = self.graph_builder.add_agent_result(
            agent_result,
            run_id=run_id,
        )

        verification_results = [
            self.fact_checker.verify_finding(finding_id)
            for finding_id in finding_ids
        ]

        structurally_verified = (
            bool(verification_results)
            and all(
                check.get("structural_verified", False) is True
                for check in verification_results
            )
        )

        verification_summary = {
            "verified": False,
            "structural_verified": structurally_verified,
            "verification_scope": "structural_only",
            "scientific_support_assessed": False,
            "checks": {
                "all_findings_structurally_verified":
                    structurally_verified,
            },
        }

        evaluation = self.evolution_engine.evaluate_execution(
            agent_result,
            verification_summary,
        )

        self.evolution_engine.update_knowledge(
            agent_result,
            evaluation,
            finding_ids,
            run_id=run_id,
        )

        return (
            finding_ids,
            verification_results,
            evaluation,
        )

    def _process_full_sequential_workflow(
        self,
        user_request,
        run_id,
        target_column=None,
    ):
        """Run the cleaning -> EDA -> optional ML/XAI -> report flow.

        Every analytical stage is recorded in the same knowledge graph. The
        ML stage hands its exact in-memory fitted model and held-out partition
        directly to ExplainabilityAgent; no second reference model is fitted.
        """
        results = []
        errors = []
        insights = []
        structural_checks = []
        cleaned_path = None

        def record_stage(stage_name, agent_result, details, stage_run_id):
            finding_ids, verification, evaluation = self._verify_and_record(
                agent_result, stage_run_id
            )
            self.knowledge_graph.save()
            results.append({
                "task_id": agent_result.task,
                "task": stage_name,
                "agent": agent_result.agent,
                "analysis_result": details,
                "findings": list(agent_result.findings),
                "finding_ids": finding_ids,
                "verification": verification,
                "evaluation": evaluation,
            })
            insights.extend(agent_result.findings)
            structural_checks.extend(verification)
            return finding_ids, verification, evaluation

        # Stage 1: safe data cleaning and provenance.
        try:
            cleaning_result, cleaning_details, cleaned_path = self.cleaning_agent.run(
                str(self.dataset_path),
                output_root=PROJECT_ROOT / "outputs" / "cleaned_datasets",
            )
            record_stage(
                "Stage 1 — Data cleaning",
                cleaning_result,
                cleaning_details,
                f"{run_id}:cleaning",
            )
        except (ValueError, OSError, TypeError) as exc:
            return self._response(
                user_request,
                "SequentialWorkflow",
                "completed_with_errors",
                message="The sequential workflow stopped during data cleaning.",
                dataset=self.dataset_path.name,
                errors=[{"stage": "DataCleaningAgent", "error": str(exc)}],
                run_id=run_id,
            )

        # Stage 2: EDA operates on the cleaned artifact, while preserving
        # missingness so EDA can report it. ML imputers are fitted later on
        # training data only.
        try:
            pipeline = run_analysis(
                str(cleaned_path),
                evolution_engine=self.evolution_engine,
            )
            for raw_result in pipeline.get("results", []):
                if "error" in raw_result:
                    errors.append({
                        "stage": "EDAAgent",
                        "task": raw_result.get("task"),
                        "error": raw_result.get("error"),
                    })
                    continue
                agent_result = analysis_result_to_agent_result(
                    raw_result,
                    dataset=self.dataset_path.name,
                    agent="EDAAgent",
                )
                task_id = raw_result.get("task_id") or agent_result.task
                record_stage(
                    f"Stage 2 — EDA: {task_id}",
                    agent_result,
                    raw_result,
                    f"{run_id}:eda:{task_id}",
                )
            if pipeline.get("insights"):
                insights.extend(pipeline["insights"])

            has_eda_result = any(
                item.get("agent") == "EDAAgent"
                for item in results
            )
            eda_failed = any(
                item.get("stage") == "EDAAgent"
                for item in errors
            )
            if not has_eda_result and not eda_failed:
                profile = pipeline.get("profile", {})
                rows = int(profile.get("n_rows", 0))
                columns = list(profile.get("columns", []))
                summary_result = AgentResult(
                    agent="EDAAgent",
                    dataset=self.dataset_path.name,
                    task="eda_summary",
                    findings=[
                        f"EDA inspected {rows} rows and {len(columns)} columns; "
                        "no findings crossed the current task-planning thresholds."
                    ],
                    features=[str(column) for column in columns],
                    evidence=[Evidence(
                        description="EDA profile and threshold-based task-planning summary",
                        method="EDA profile summary",
                        value={
                            "rows": rows,
                            "columns": len(columns),
                            "ranked_insights": len(pipeline.get("insights", [])),
                            "planned_tasks": len(pipeline.get("tasks", [])),
                        },
                        finding_indices=[0],
                    )],
                    confidence=estimate_confidence(rows),
                )
                record_stage(
                    "Stage 2 — EDA summary (no threshold-triggered tasks)",
                    summary_result,
                    {
                        "agent": "EDAAgent",
                        "task": "eda_summary",
                        "profile": {
                            "rows": rows,
                            "columns": len(columns),
                            "ranked_insights": len(pipeline.get("insights", [])),
                            "planned_tasks": len(pipeline.get("tasks", [])),
                        },
                    },
                    f"{run_id}:eda:summary",
                )
        except (ValueError, OSError, TypeError) as exc:
            errors.append({"stage": "EDAAgent", "error": str(exc)})

        # Stages 3-4: optional supervised learning and explanation. Passing no
        # target is a supported EDA-only path for unsupervised datasets.
        if target_column is not None and str(target_column).strip():
            target_column = str(target_column).strip()
            model_artifacts = None
            try:
                ml_result, ml_details, model_artifacts = self.ml_agent.run(
                    str(cleaned_path),
                    target_column=target_column,
                    return_artifacts=True,
                )
                record_stage(
                    "Stage 3 — ML training and evaluation",
                    ml_result,
                    ml_details,
                    f"{run_id}:ml:{target_column}",
                )
            except (ValueError, OSError, TypeError) as exc:
                errors.append({"stage": "MLAgent", "error": str(exc)})

            if model_artifacts is not None:
                try:
                    xai_result, xai_details = self.explainability_agent.explain_fitted_model(
                        model_artifacts,
                        dataset_name=self.dataset_path.name,
                        target_column=target_column,
                    )
                    record_stage(
                        "Stage 4 — Explainability of fitted model",
                        xai_result,
                        xai_details,
                        f"{run_id}:xai:{target_column}",
                    )
                except (ValueError, OSError, TypeError) as exc:
                    errors.append({"stage": "ExplainabilityAgent", "error": str(exc)})
        else:
            results.append({
                "task_id": "prediction_skipped",
                "task": "Stage 3–4 — Predictive modeling skipped",
                "agent": "SequentialWorkflow",
                "analysis_result": {
                    "reason": "No target column was selected. Cleaning and EDA ran; ML and model explainability were skipped.",
                },
                "findings": [
                    "No target selected; the workflow completed data cleaning and EDA without fitting a predictive model."
                ],
                "finding_ids": [],
                "verification": [],
                "evaluation": None,
            })

        # Stage 5: report from the graph after all completed analysis stages.
        self.knowledge_graph.save()
        try:
            report_details = self.report_agent.run(
                self.knowledge_graph,
                dataset_name=self.dataset_path.name,
                title="Sequential Explainable Analytics Report",
            )
            results.append({
                "task_id": "final_report",
                "task": "Stage 5 — Final report",
                "agent": "ReportAgent",
                "analysis_result": report_details,
                "findings": [report_details.get("summary", "Final report generated.")],
                "finding_ids": [],
                "verification": [],
                "evaluation": None,
            })
        except (ValueError, OSError, TypeError) as exc:
            errors.append({"stage": "ReportAgent", "error": str(exc)})

        executed_stages = sum(
            1 for item in results if item.get("evaluation") is not None
        )
        all_verified = bool(structural_checks) and all(
            item.get("structural_verified") is True
            for item in structural_checks
        )
        if errors:
            status = "completed_with_errors"
            message = (
                "The sequential workflow finished with one or more stage errors. "
                "Review the errors listed with the workflow results."
            )
        elif all_verified:
            status = "completed"
            if target_column:
                message = (
                    "Sequential workflow completed: cleaning → EDA → ML → "
                    "explainability of the fitted model → report."
                )
            else:
                message = "Sequential workflow completed: cleaning → EDA → report. Predictive stages were skipped because no target was selected."
        else:
            status = "completed_with_warnings"
            message = "The sequential workflow completed, but at least one stage did not provide fully passing structural checks."

        return self._response(
            user_request,
            "SequentialWorkflow",
            status,
            message=message,
            dataset=self.dataset_path.name,
            task_count=executed_stages,
            insights=insights,
            results=results,
            errors=errors,
            run_id=run_id,
        )

    def process_request(
        self,
        user_request,
        run_id=None,
        target_column=None,
    ):
        """Route a request and execute its selected agent."""

        if (
            not isinstance(user_request, str)
            or not user_request.strip()
        ):
            raise ValueError(
                "user_request must be a non-empty string."
            )

        user_request = user_request.strip()

        if run_id is None:
            run_id = uuid4().hex
        elif (
            not isinstance(run_id, str)
            or not run_id.strip()
        ):
            raise ValueError(
                "run_id must be a non-empty string."
            )
        else:
            run_id = run_id.strip()

        if (
            "sequential analytics workflow" in user_request.lower()
            or "full sequential workflow" in user_request.lower()
        ):
            return self._process_full_sequential_workflow(
                user_request,
                run_id,
                target_column=target_column,
            )

        selected_agent = self.router.route(user_request)

        # ---------------------------------------------------------
        # DATA AGENT
        # ---------------------------------------------------------
        if selected_agent == "DataAgent":
            try:
                agent_result, profile = self.data_agent.run(
                    str(self.dataset_path)
                )
            except (ValueError, OSError) as exc:
                return self._response(
                    user_request,
                    selected_agent,
                    "completed_with_errors",
                    message="Data profiling could not be completed.",
                    dataset=self.dataset_path.name,
                    errors=[{"error": str(exc)}],
                    run_id=run_id,
                )

            task_run_id = f"{run_id}:{agent_result.task}"

            (
                finding_ids,
                verification_results,
                evaluation,
            ) = self._verify_and_record(
                agent_result,
                task_run_id,
            )

            self.knowledge_graph.save()

            structurally_verified = (
                bool(verification_results)
                and all(
                    item.get("structural_verified", False) is True
                    for item in verification_results
                )
            )

            status = (
                "completed"
                if structurally_verified
                else "completed_with_warnings"
            )

            return self._response(
                user_request,
                selected_agent,
                status,
                message="Data quality profiling completed.",
                dataset=agent_result.dataset,
                task_count=1,
                insights=agent_result.findings,
                results=[
                    {
                        "task_id": agent_result.task,
                        "task": agent_result.task,
                        "analysis_result": profile,
                        "findings": agent_result.findings,
                        "finding_ids": finding_ids,
                        "verification": verification_results,
                        "evaluation": evaluation,
                    }
                ],
                run_id=run_id,
            )

        # ---------------------------------------------------------
        # ML AGENT
        # ---------------------------------------------------------
        if selected_agent == "MLAgent":
            if (
                not isinstance(target_column, str)
                or not target_column.strip()
            ):
                return self._response(
                    user_request,
                    selected_agent,
                    "input_required",
                    message=(
                        "MLAgent requires a target column. "
                        "Specify the column you want the model "
                        "to predict."
                    ),
                    dataset=self.dataset_path.name,
                    run_id=run_id,
                )

            target_column = target_column.strip()

            try:
                agent_result, model_details = self.ml_agent.run(
                    str(self.dataset_path),
                    target_column=target_column,
                )
            except (ValueError, OSError) as exc:
                return self._response(
                    user_request,
                    selected_agent,
                    "completed_with_errors",
                    message=(
                        "ML classification could not be completed."
                    ),
                    dataset=self.dataset_path.name,
                    errors=[{"error": str(exc)}],
                    run_id=run_id,
                )

            task_run_id = (
                f"{run_id}:ML:{target_column}"
            )

            (
                finding_ids,
                verification_results,
                evaluation,
            ) = self._verify_and_record(
                agent_result,
                task_run_id,
            )

            self.knowledge_graph.save()

            structurally_verified = (
                bool(verification_results)
                and all(
                    item.get("structural_verified", False) is True
                    for item in verification_results
                )
            )

            status = (
                "completed"
                if structurally_verified
                else "completed_with_warnings"
            )

            return self._response(
                user_request,
                selected_agent,
                status,
                message=(
                    "ML classification and evaluation completed. "
                    "Metrics describe held-out test performance."
                ),
                dataset=agent_result.dataset,
                task_count=1,
                insights=agent_result.findings,
                results=[
                    {
                        "task_id": agent_result.task,
                        "task": agent_result.task,
                        "analysis_result": model_details,
                        "findings": agent_result.findings,
                        "finding_ids": finding_ids,
                        "verification": verification_results,
                        "evaluation": evaluation,
                    }
                ],
                run_id=run_id,
            )

        # ---------------------------------------------------------
        # EXPLAINABILITY AGENT
        # ---------------------------------------------------------
        if selected_agent == "ExplainabilityAgent":
            if (
                not isinstance(target_column, str)
                or not target_column.strip()
            ):
                return self._response(
                    user_request,
                    selected_agent,
                    "input_required",
                    message=(
                        "ExplainabilityAgent requires a target column. "
                        "Specify the classification outcome whose "
                        "predictions you want to explain."
                    ),
                    dataset=self.dataset_path.name,
                    run_id=run_id,
                )

            target_column = target_column.strip()

            try:
                agent_result, explanation_details = (
                    self.explainability_agent.run(
                        str(self.dataset_path),
                        target_column=target_column,
                    )
                )
            except (ValueError, OSError, TypeError) as exc:
                return self._response(
                    user_request,
                    selected_agent,
                    "completed_with_errors",
                    message=(
                        "Model explanation could not be completed."
                    ),
                    dataset=self.dataset_path.name,
                    errors=[{"error": str(exc)}],
                    run_id=run_id,
                )

            task_run_id = (
                f"{run_id}:EXPLAINABILITY:{target_column}"
            )

            (
                finding_ids,
                verification_results,
                evaluation,
            ) = self._verify_and_record(
                agent_result,
                task_run_id,
            )

            self.knowledge_graph.save()

            structurally_verified = (
                bool(verification_results)
                and all(
                    item.get("structural_verified", False) is True
                    for item in verification_results
                )
            )

            status = (
                "completed"
                if structurally_verified
                else "completed_with_warnings"
            )

            return self._response(
                user_request,
                selected_agent,
                status,
                message=(
                    "Permutation feature-importance analysis completed "
                    "on held-out test data. The explanation uses a "
                    "newly fitted reference model."
                ),
                dataset=agent_result.dataset,
                task_count=1,
                insights=agent_result.findings,
                results=[
                    {
                        "task_id": agent_result.task,
                        "task": agent_result.task,
                        "analysis_result": explanation_details,
                        "findings": agent_result.findings,
                        "finding_ids": finding_ids,
                        "verification": verification_results,
                        "evaluation": evaluation,
                    }
                ],
                run_id=run_id,
            )

        # ---------------------------------------------------------
        # REPORT AGENT
        # ---------------------------------------------------------
        if selected_agent == "ReportAgent":
            try:
                report = self.report_agent.run(
                    self.knowledge_graph,
                    dataset_name=self.dataset_path.name,
                    title="Data Analytics Report",
                )
            except (TypeError, ValueError, OSError) as exc:
                return self._response(
                    user_request,
                    selected_agent,
                    "completed_with_errors",
                    message="Report generation could not be completed.",
                    dataset=self.dataset_path.name,
                    errors=[{"error": str(exc)}],
                    run_id=run_id,
                )

            report_status = (
                "completed"
                if report["status"] == "completed"
                else "no_findings"
            )

            report_findings = report["findings"]

            return self._response(
                user_request,
                selected_agent,
                report_status,
                message=report["summary"],
                dataset=self.dataset_path.name,
                task_count=1,
                insights=[
                    finding["description"]
                    for finding in report_findings
                ],
                results=[
                    {
                        "task_id": "generate_report",
                        "task": "generate_report",
                        "analysis_result": report,
                        "findings": [
                            finding["description"]
                            for finding in report_findings
                        ],
                        "finding_ids": [
                            finding["finding_id"]
                            for finding in report_findings
                        ],
                        "verification": [],
                        "evaluation": None,
                    }
                ],
                run_id=run_id,
            )

        # ---------------------------------------------------------
        # UNSUPPORTED AGENT FALLBACK
        # ---------------------------------------------------------
        if selected_agent != "EDAAgent":
            return self._response(
                user_request,
                selected_agent,
                "not_implemented",
                message=(
                    f"{selected_agent} has been selected, but its "
                    "execution workflow is not yet connected."
                ),
                dataset=self.dataset_path.name,
                run_id=run_id,
            )

        # ---------------------------------------------------------
        # EDA AGENT
        # ---------------------------------------------------------
        try:
            pipeline = run_analysis(
                str(self.dataset_path),
                evolution_engine=self.evolution_engine,
            )
        except (ValueError, OSError) as exc:
            return self._response(
                user_request,
                selected_agent,
                "completed_with_errors",
                message="EDA workflow could not be completed.",
                dataset=self.dataset_path.name,
                errors=[{"error": str(exc)}],
                run_id=run_id,
            )

        converted_results = []
        failed_results = []

        for raw_result in pipeline["results"]:
            if "error" in raw_result:
                failed_results.append(raw_result)
                continue

            agent_result = analysis_result_to_agent_result(
                raw_result,
                dataset=self.dataset_path.name,
                agent=selected_agent,
            )

            task_id = (
                raw_result.get("task_id")
                or agent_result.task
            )

            task_run_id = f"{run_id}:{task_id}"

            (
                finding_ids,
                verification_results,
                evaluation,
            ) = self._verify_and_record(
                agent_result,
                task_run_id,
            )

            self.knowledge_graph.save()

            converted_results.append(
                {
                    "task_id": raw_result.get("task_id"),
                    "task": raw_result.get("task"),
                    "analysis_result": raw_result,
                    "findings": agent_result.findings,
                    "finding_ids": finding_ids,
                    "verification": verification_results,
                    "evaluation": evaluation,
                }
            )

        all_structurally_verified = (
            bool(converted_results)
            and not failed_results
            and all(
                check.get("structural_verified", False) is True
                for item in converted_results
                for check in item["verification"]
            )
        )

        if failed_results:
            status = "completed_with_errors"
        elif all_structurally_verified:
            status = "completed"
        else:
            status = "completed_with_warnings"

        return self._response(
            user_request,
            selected_agent,
            status,
            message="EDA workflow executed.",
            dataset=pipeline.get(
                "dataset",
                self.dataset_path.name,
            ),
            task_count=len(pipeline["tasks"]),
            insights=pipeline["insights"],
            results=converted_results,
            errors=failed_results,
            run_id=run_id,
        )


if __name__ == "__main__":
    orchestrator = Orchestrator()

    request = input("What would you like to analyze? ")
    result = orchestrator.process_request(request)

    print("\nRequest:", result["user_request"])
    print("Run ID:", result["run_id"])
    print("Selected agent:", result["selected_agent"])
    print("Status:", result["status"])
    print("Message:", result["message"])

    if result["status"] == "completed":
        print("Tasks executed:", result["task_count"])
        print(
            "Findings recorded:",
            sum(
                len(item.get("finding_ids", []))
                for item in result["results"]
            ),
        )
