# adaptive-multi-agent-data-analysis

adaptive-ai-data-analysis/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── app/
│
│   ├── orchestrator/
│   │   ├── orchestrator.py
│   │   └── router.py
│   │
│   ├── agents/
│   │   ├── data_agent.py
│   │   ├── eda_agent.py
│   │   ├── ml_agent.py
│   │   └── report_agent.py
│   │
│   ├── knowledge_graph/
│   │   ├── graph.py
│   │   ├── nodes.py
│   │   └── reasoning.py
│   │
│   ├── verification/
│   │   └── fact_checker.py
│   │
│   └── tools/
│       ├── data_tools.py
│       ├── ml_tools.py
│       └── visualization_tools.py
│
├── data/
├── reports/
├── tests/
└── docs/



                 main
                  │
        ┌─────────┼─────────┐
        ↓         ↓         ↓
   Member 1   Member 2    YOU
   branch      branch     branch
        │         │         │
 Orchestrator   Agents      KG
        │         │         │
        └─────────┼─────────┘
                  ↓
              Pull Request
                  ↓
                 main


An adaptive multi-agent AI system for automated data analysis, knowledge-grounded reasoning, verification, and explainable report generation using a Dynamic Knowledge Graph.

---

## 📌 Overview

Traditional data analysis often requires users to manually perform data cleaning, exploratory analysis, machine learning, visualization, and interpretation.

This project proposes an **Adaptive Multi-Agent AI Data Analysis System** that automatically coordinates specialized AI agents to analyze datasets and generate an explainable report.

The system combines:

- Multi-Agent AI
- Adaptive Task Routing
- Automated Data Analysis
- Machine Learning
- Explainable AI
- Dynamic Knowledge Graphs
- Graph-Based Reasoning
- Fact Checking and Verification
- Automated Report Generation

The goal is to create a system that not only produces analytical results but also explains **how the results were obtained, what evidence supports them, and how confident the system is in its conclusions.**

---

## 🎯 Objectives

The main objectives of this project are:

1. Develop an adaptive multi-agent framework for automated data analysis.
2. Automatically decompose user requests into analytical tasks.
3. Dynamically select suitable agents and tools for each task.
4. Perform automated data cleaning, EDA, visualization, and machine learning.
5. Store datasets, features, findings, models, and evidence in a Dynamic Knowledge Graph.
6. Use graph-based reasoning to connect findings with supporting evidence.
7. Verify analytical findings using a Critic / Fact Checker module.
8. Generate explainable reports containing findings, evidence, visualizations, confidence scores, and Knowledge Graph references.

---

## 🏗️ System Architecture

```text
                         USER
                           │
                           ▼
                   ┌───────────────┐
                   │ ORCHESTRATOR  │
                   └───────┬───────┘
                           │
                           ▼
                   ┌───────────────┐
                   │ ADAPTIVE      │
                   │ ROUTER        │
                   └───────┬───────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
         DATA AGENT     EDA AGENT     ML AGENT
              │            │            │
              └────────────┼────────────┘
                           ▼
                 DYNAMIC KNOWLEDGE
                       GRAPH
                           │
                           ▼
                  GRAPH REASONING
                           │
                           ▼
                 CRITIC / FACT CHECKER
                           │
                           ▼
                    REPORT GENERATOR
                           │
                           ▼
                EXPLAINABLE AI REPORT

---

## Context-aware analytical memory (research prototype)

The EDA correlation workflow now builds a versioned context profile from the input column pair (sample size, missingness, skewness, IQR-outlier fraction, Pearson/Spearman coefficient gap, and constant-column indicators). This profile is saved with evidence and the execution node in the knowledge graph.

When a context is supplied, the strategy recommender retrieves structurally successful historical executions with similar profiles, uses similarity-weighted strategy votes, groups repeated instances of the same dataset/column/profile context to reduce repeat-run bias, and abstains when the vote is weak or tied. If no suitable context is available, the EDA workflow falls back to its data-based heuristic. The legacy task-level strategy recommender remains available when no context is supplied.

The dashboard uses a stable dataset fingerprint for the knowledge-graph file and a content-addressed upload path so a later run can reload that dataset's graph. Old session-specific graph files are not automatically migrated.

### Research status and limitations

This is a research prototype, not yet evidence of statistical superiority. The current execution success flag primarily describes structural verification and execution-policy checks; it does not prove that a selected statistical method is scientifically optimal. The research plan in [`research/CONFERENCE_RESEARCH_PLAN.md`](research/CONFERENCE_RESEARCH_PLAN.md) defines the next evaluation stage, baselines, ablations, and reproducibility requirements.

Run the full tests with:

```powershell
python -m pytest -q
```

## Reproducible strategy-selection benchmark

The research scripts compare fixed Pearson, fixed Spearman, the existing skew heuristic, task-majority voting with heuristic fallback, and context-aware historical memory on six controlled synthetic scenario families. The output contains CSV/JSON case-level records, aggregate summaries, plots, a duplicate-history stress test, and repeated-seed reports.

Run a single benchmark:

```powershell
python -m research.benchmark_strategy_selection --train-per-family 12 --test-per-family 30 --seed 20261025 --output outputs/research_evaluation
```

Repeat across five seeds:

```powershell
python -m research.run_repeated_benchmark --seeds 20261025 20261026 20261027 20261028 20261029 --train-per-family 12 --test-per-family 30 --output outputs/research_evaluation/multi_seed
```

The initial controlled experiment evaluated 900 held-out synthetic queries over five seeds and reported 97.7% mean strategy-selection accuracy for contextual memory vs 82.0% for the skew heuristic. This is a preliminary synthetic result, not a claim of real-world superiority: the training memory uses oracle-labelled scenario objectives and the scenario families are represented in both training and testing. Details, limitations, and next experiments are documented in [`research/CONFERENCE_RESEARCH_PLAN.md`](research/CONFERENCE_RESEARCH_PLAN.md) and [`outputs/research_evaluation/multi_seed/repeated_seed_report.md`](outputs/research_evaluation/multi_seed/repeated_seed_report.md).

## Leave-one-synthetic-family-out evaluation

The next validation step removes an entire scenario family from the experience memory in each fold and tests contextual strategy selection only on that held-out family. Each fold uses a new graph, and only the remaining five scenario families are seeded into memory. This is stricter than the repeated-seed benchmark, although it is still synthetic and uses predeclared scenario reference labels.

Run the five-seed protocol:

```powershell
python -m research.run_leave_one_family_out --seeds 20261025 20261026 20261027 20261028 20261029 --train-per-family 12 --test-per-family 30 --output outputs/research_evaluation/leave_one_family_out
```

The run produces `leave_one_family_out_report.md`, per-method/per-family/per-seed CSVs, a case-level CSV, JSON output, and an accuracy plot. The run executed with 900 test queries across 30 folds. Contextual memory averaged 84.0% accuracy (SD 1.7 percentage points across seeds) versus 80.8% (SD 0.8 pp) for the skew heuristic. These results are preliminary: family-level folds reuse overlapping training scenarios; oracle labels are used for the known families; only six synthetic families are included; and the task-majority baseline is mechanically misleading under this fold design because excluding one family makes the held-out class the minority among the remaining five families. Do not interpret the task-majority 0% score as a realistic expectation for general task-level memory.

The per-family result matters: contextual memory reaches 42.0% for the outlier-contaminated monotonic family, compared with 24.7% for the skew heuristic, but reaches 90.0% for the clean linear family, compared with 100.0% for the heuristic. The average result therefore does not imply improvement on every scenario. The next steps remain independently sourced real datasets, stronger out-of-distribution abstention tests, and ablations.


## Leave-one-scenario-family-out baseline definitions

The LOFO report separates two historical baselines: `dataset_history_fallback` follows the application's task-and-dataset isolation, so a newly named held-out dataset has no exact-dataset history and falls back to the skew heuristic; `global_majority_heuristic_fallback` is a naive cross-dataset vote included only as a diagnostic. Its 0% accuracy in this synthetic LOFO setup is induced by the 3-vs-3 reference-label design and should not be interpreted as a general property of historical-memory systems. See `outputs/research_evaluation/leave_one_family_out/leave_one_family_out_report.md`.

## Real-data perturbation robustness evaluation

The next evaluation layer uses multiple bundled real tabular datasets and evaluates whether selected correlation coefficients remain stable under controlled missingness and extreme-outlier perturbations. It uses a leave-one-dataset-out protocol: experiences for each fold come only from the other datasets.

Run a small smoke evaluation:

```powershell
python -m research.run_real_dataset_robustness --seeds 20261025 --max-pairs-per-dataset 8 --perturbation-replicates 2 --output outputs/research_evaluation/real_dataset_robustness_smoke
```

Run the fuller reproducible experiment:

```powershell
python -m research.run_real_dataset_robustness --seeds 20261025 20261026 20261027 --max-pairs-per-dataset 20 --perturbation-replicates 2 --output outputs/research_evaluation/real_dataset_robustness
```

This experiment evaluates a narrow objective: which method's perturbed coefficient has a smaller absolute shift from that same method's clean-data estimate under a declared perturbation family. It does not identify scientific truth, does not establish that Pearson or Spearman is universally superior, and must not be described as external validation of universal method correctness. The report documents data sources, held-out protocol, target definition, ambiguity margin, coverage, and limitations.

Dataset sources and citations are recorded in `research/run_real_dataset_robustness.py` and in the generated report. The default evaluation excludes the project-provided Titanic sample because its upstream provenance is not documented. `--include-titanic-sample` is available for exploratory checks only until the exact source/version/license are verified.


The current real-data perturbation pilot is a deliberately narrow robustness test, not scientific ground truth. In the verified run (three seeds, four attributed datasets, 66 feature pairs, 792 queries), contextual memory scored 83.3% against the stability-derived target, while fixed Spearman/task-majority reached 84.6%. This is an important limitation: contextual memory has not demonstrated an advantage over the simpler fixed-Spearman baseline for this objective. See `outputs/research_evaluation/real_dataset_robustness/real_dataset_robustness_report.md` and the expanded `research/PAPER_RESULTS_DRAFT.md` for methods and caveats.

## Full sequential analytics workflow

The dashboard includes a **Full sequential analytics workflow** action. After uploading/selecting a CSV and clicking Run, the orchestrator executes these stages in order:

1. **DataCleaningAgent** writes a separate, content-addressed cleaned CSV artifact, trims surrounding whitespace from column names, removes exact duplicate rows and entirely empty columns, and converts infinite numeric values to missing values. It records the cleaning actions and manifest. It deliberately does **not** impute all missing values globally; missingness remains available to EDA, while the ML pipeline fits imputers only on the training split.
2. **EDA** runs on the cleaned artifact, records correlation/missingness/outlier findings, and can use the context-aware strategy recommender.
3. **MLAgent (optional)** runs when a target column is selected, trains/evaluates a classification pipeline, and keeps the fitted estimator and held-out partition in memory for the next stage.
4. **ExplainabilityAgent (optional)** computes permutation importance using the **same fitted model and held-out partition produced by MLAgent**. It does not refit a separate model in this sequential flow.
5. **ReportAgent** generates a Markdown report from the knowledge graph after the completed stages.

If no target is selected, the workflow runs cleaning, EDA, and report generation while skipping the supervised ML/XAI stages. Individual focused-agent actions remain available. Uploading a changed CSV changes its content fingerprint; the sequential workflow starts when the user explicitly clicks the Run button—there is no background filesystem watcher or automatic execution on file changes.

The cleaning and modeling steps are intentionally leakage-aware: the cleaning stage preserves missingness for analysis and does not calculate global imputation values. The ML preprocessing pipeline fits numeric and categorical imputers as part of the model pipeline using training data only. Graph verification remains structural and metadata-focused; it should not be described as a proof that every analytical conclusion is scientifically true.


## NEXUS AI UI/UX workspace

The dashboard now has a refreshed visual hierarchy and eight purpose-specific views:

- **Overview:** dataset/run summary and core capability cards.
- **Workflow:** an execution-aware CSV → cleaning → EDA → optional ML → explainability → verification → report flowchart. Stage status is based on the latest recorded result; stages are not marked complete merely because they appear in the diagram.
- **Data explorer:** bounded CSV preview (up to 10,000 rows), per-column types, missingness, unique values, cleaning log, and downloadable cleaned copy when one was produced.
- **Visual analytics:** missingness chart, numeric distribution histogram, Pearson heatmap, IQR outlier overview, and feature-importance chart when a model explanation exists.
- **Findings & XAI:** finding-level evidence, selected method, rationale, and verification notes.
- **Knowledge graph:** focus a node to view a local subgraph, node metadata, and direct relationships.
- **Memory & history:** current correlation strategy decisions, data-context features, recorded strategy counts, and execution history.
- **Reports & evaluation:** latest workflow report, saved research documents, and existing benchmark summary artifacts.

### Run the enhanced dashboard

```powershell
python -m pytest -q
python -m streamlit run dashboard.py
```

Upload a CSV (or use the included Titanic sample), choose **Full sequential analytics workflow**, and run it. Select a target such as `Survived` to include the ML and explainability stages; choose the skip option to run cleaning, EDA, and reporting only. Uploading a changed file creates a different content fingerprint, but analysis is started explicitly by clicking **Run selected workflow**—there is no background file watcher.

The dataset explorer reads a bounded preview for responsiveness. The full workflow uses the cleaning and analysis pipeline for its actual results. EDA correlation heatmaps in the visual workspace are a descriptive Pearson overview; individual strategy decisions can choose Pearson or Spearman and show their own rationale.
