# NEXUS AI UI/UX Guide

## What changed

The refreshed dashboard groups work into eight tabs and adds a flowchart plus data-inspection panels. It is still a Streamlit/Python dashboard; no separate frontend framework was introduced.

### 1. Overview
A clearer landing area, dataset label, graph-ready status, KPI cards, and core-capability cards.

### 2. Workflow
The graph visualizes CSV input, cleaning, EDA, optional model training, explainability, verification, and report generation. Status text comes from the latest output record. For focused workflows, unrelated stages are marked as not part of that run.

### 3. Data explorer
Displays up to 10,000 rows from the selected input or cleaned artifact, column types, missing percentages, unique values, and the cleaning log if a full sequential run has created one. The preview is bounded; it is not represented as a full-dataset row count when truncation applies.

### 4. Visual analytics
Provides missingness bars, a numeric distribution, a Pearson correlation heatmap, IQR outlier counts, and a feature-importance view when explainability results are present. Visuals calculated from the preview are labelled as such.

### 5. Findings & XAI
Keeps finding evidence, method, rationale, structural verification, and relevant model metadata together. The fact checker’s scope is stated plainly: structural verification does not prove scientific truth.

### 6. Knowledge graph
Allows selection of a graph node, renders a focused neighbourhood, shows that node’s attributes, and lists its incoming/outgoing relationships.

### 7. Memory & history
Surfaces the current run’s correlation decisions with the decision source, profile indicators, and rationale, plus saved execution contexts and strategy frequencies. Historical frequency/support are advisory signals, not proof that a method is statistically superior.

### 8. Reports & evaluation
Displays the latest generated report, existing research reports, and saved benchmark summaries/score charts. The dashboard does not silently rerun research experiments.

## Run and verify

From the project root:

```powershell
python -m pytest -q
python -m streamlit run dashboard.py
```

The full sequential workflow is explicitly initiated with the Run button. A changed CSV receives a new content fingerprint; there is no automatic background file watcher. A target is optional for the full flow—without one, ML and fitted-model explainability are skipped.
