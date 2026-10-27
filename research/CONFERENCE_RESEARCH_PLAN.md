# Research Plan: Context-Aware, Evidence-Grounded Analytical Memory

## Status

This is a research plan and implementation note, not a claim that novelty or empirical superiority has already been established.

The first research-oriented improvement is implemented in the current prototype: correlation analyses create a versioned data-context profile, the knowledge graph stores that profile with the execution record, and the recommender retrieves contextually similar historical experiences rather than relying only on task-level strategy frequency.

## Candidate research question

Does retrieving contextually similar analytical experiences improve correlation-strategy selection over fixed strategies, a data-profile heuristic, and task-level historical majority voting on held-out datasets and synthetic scenarios with known properties?

## Current implementation

### Context profile (version 1)

For each numeric column pair, `src.analysis.build_correlation_context` records:

- log-scaled complete sample size;
- missing-row fraction;
- maximum and mean absolute skewness across the two columns;
- maximum IQR-outlier fraction;
- absolute Pearson-versus-Spearman coefficient gap;
- whether that gap could be calculated; and
- fraction of columns that are constant.

The context is stored separately from the finding text and selected method. This allows the recommender to compare input characteristics, rather than retrieve a prior output solely because its label matches.

### Retrieval and recommendation

`SelfEvolutionEngine.recommend_strategy(..., context=...)` uses a weighted normalized numeric-profile distance. The current prototype:

1. considers executions of the same task that the existing evaluation marked successful and that contain a compatible context profile;
2. excludes contexts below the configured similarity threshold (default 0.60);
3. groups repeated executions with the same dataset, column pair, and rounded context profile so repeated clicks do not create unlimited votes;
4. takes up to five nearest distinct experiences;
5. weights strategy votes by squared context similarity; and
6. abstains when the weighted support is weak or tied, allowing the data-based heuristic to decide.

The recommendation rationale reports contextual coverage, nearest and mean similarity, and weighted support. Contextual retrieval can transfer experiences across dataset names, and exposes source datasets/columns in the returned recommendation for auditing.

## Important limitation

The existing execution `successful` flag primarily reflects structural verification, confidence thresholds, and model-metric checks when present. It does **not** establish that a correlation method is statistically optimal or scientifically correct. Therefore, the current contextual memory is a retrieval prototype, not yet a fully quality-verified self-improvement mechanism.

The next research milestone is to evaluate strategy quality independently and record that quality separately from structural validity. Only then should quality-verified outcomes drive policy updates.

## Required baselines

1. **Fixed Pearson:** always use Pearson correlation.
2. **Fixed Spearman:** always use Spearman correlation.
3. **Current data-based heuristic:** use `select_correlation_strategy` without historical memory.
4. **Task-level historical majority:** use the legacy recommender without a context profile.
5. **Context-aware memory:** use versioned data profiles and similarity-weighted retrieval, falling back to the current heuristic when evidence is insufficient.

## Experimental design

### Synthetic scenarios

Create repeatable scenarios with fixed random seeds and known data-generating processes. Include at least:

- approximately linear relationships with light-tailed noise;
- monotonic nonlinear relationships;
- skewed marginal distributions;
- extreme outliers;
- varying sample sizes; and
- missing values at controlled rates.

The research protocol must define what each scenario counts as an appropriate analysis objective before running experiments. Pearson and Spearman measure different forms of association; neither is universally superior. Scenarios should therefore test a declared objective and the expected behavior for that objective, not assume one method is always correct.

### Real datasets

Include multiple appropriately licensed tabular datasets. Hold out complete datasets (or dataset families) from experience construction to measure transfer to new data. Do not split only by rows when records from the same dataset can leak into both history and evaluation.

### Metrics

- strategy-selection accuracy where the benchmark has a defensible reference decision;
- recommendation coverage and abstention rate;
- robustness of estimates under controlled resampling and perturbations;
- context retrieval similarity and source-context traceability;
- explanation fidelity (whether the explanation matches the code path actually taken); and
- runtime and memory overhead.

Report uncertainty intervals and all denominators. Do not make a superiority claim until the measured results support it.

### Ablations

- remove skew features;
- remove outlier and missingness features;
- remove the Pearson–Spearman gap feature;
- disable duplicate-context grouping;
- disable cross-dataset retrieval; and
- compare context-aware retrieval against task-level majority voting.

## Reproducibility checklist

- Record random seeds and software versions.
- Save dataset fingerprints, context-profile version, chosen strategy, recommendation scores, fallback reason, and execution identifiers.
- Keep training/history datasets separate from held-out evaluation datasets.
- Preserve the same evaluation protocol for all baselines.
- Publish scripts and aggregate results alongside the paper when dataset licenses permit.

## Claim discipline

Potential contribution: a context-aware, evidence-linked memory mechanism for reusable analytical strategy selection that includes duplicate-context suppression and explicit abstention when retrieved evidence is ambiguous.

This is a candidate contribution, not a guarantee of global novelty. A targeted literature review and controlled experiments are required before making a novelty or performance claim.

## Initial synthetic benchmark run (25 October 2026)

A first executable benchmark was added at `research/benchmark_strategy_selection.py`, with repeated-seed execution at `research/run_repeated_benchmark.py`. The scripts call the project's actual `build_correlation_context`, `select_correlation_strategy`, and `SelfEvolutionEngine.recommend_strategy` implementation. Run them from the repository root as follows:

```powershell
python -m research.benchmark_strategy_selection --train-per-family 12 --test-per-family 30 --seed 20261025 --output outputs/research_evaluation
python -m research.run_repeated_benchmark --seeds 20261025 20261026 20261027 20261028 20261029 --train-per-family 12 --test-per-family 30 --output outputs/research_evaluation/multi_seed
```

### Protocol actually executed

- Six synthetic scenario families were defined: clean linear, linear with MCAR missingness, linear with small sample size, monotonic exponential, monotonic cubic, and an underlying linear/monotonic relation with extreme response contamination where the declared objective is robust rank association.
- Each family is represented in both the training-memory pool and held-out test queries, but with disjoint random seeds. Sample sizes vary approximately ±15% from each family base; noise and relevant missingness/contamination levels are randomized. This is **held-out-sample evaluation within known scenario families**, not leave-one-family-out or real-domain generalization.
- The reference method is part of the predeclared synthetic scenario specification. Each seed has 72 oracle-labelled training experiences and 180 held-out queries; five seeds produce 900 test queries total.
- Baselines are fixed Pearson, fixed Spearman, the implemented skew heuristic, task-majority voting with the application's abstain-to-heuristic fallback, and contextual memory. In the balanced training pool the global strategy counts tie, so that task-majority baseline correctly falls back to the skew heuristic; the benchmark reports this behavior rather than pretending it made an independent majority decision.
- A separate duplicate-history stress test adds 20 copies of one Pearson execution context. Raw global counts change from a tie (36 Pearson / 36 Spearman) to Pearson-majority (56 / 36), while the contextual recommendation and its support remain unchanged because the copies share the same dataset, column pair, and rounded profile. The replay copies are not independent observations.

### Preliminary results

Across five seeds (900 held-out synthetic queries), mean selection accuracy was:

| Method | Mean accuracy | SD across seeds | 95% t interval across seeds |
|---|---:|---:|---:|
| Fixed Pearson | 50.0% | 0.0 pp | [50.0%, 50.0%] |
| Fixed Spearman | 50.0% | 0.0 pp | [50.0%, 50.0%] |
| Skew heuristic | 82.0% | 0.3 pp | [81.6%, 82.4%] |
| Task majority + heuristic fallback | 82.0% | 0.3 pp | [81.6%, 82.4%] |
| Contextual memory | 97.7% | 0.2 pp | [97.4%, 98.0%] |

The mean paired seed-level accuracy difference between contextual memory and the skew heuristic was +15.7 percentage points (95% t interval [15.1, 16.2] pp across five seeds). The contextual method had 100% recommendation coverage within this synthetic protocol. Per-scenario averages indicate the largest gap in the outlier-contaminated monotonic scenario (contextual memory 100.0% vs skew heuristic 22.7%) and an improvement for cubic monotonic data (87.3% vs 76.0%). See `outputs/research_evaluation/multi_seed/repeated_seed_report.md` and `repeated_seed_per_scenario.csv` for results and supporting data.

### Interpretation boundary

These results are preliminary and synthetic. Training memory was seeded with oracle-labelled scenario objectives, not labels inferred from the application's structural-verification pass flag. The scenario families are present in both splits; therefore these measurements do not demonstrate generalization to unseen families, held-out real datasets, or autonomous quality discovery. The balanced strategy prior also means the task-majority-with-fallback baseline coincides with the skew heuristic in this protocol. The observed advantage supports continuing evaluation; it does not establish global novelty or statistical superiority on real-world data.

### Required next experiments before a conference claim

1. Hold out entire scenario families during experience construction and test contextual transfer to an unseen family.
2. Add at least several independently sourced/licensed real datasets, with a strategy-quality rubric defined independently of the method under test.
3. Ablate individual profile features, duplicate-context grouping, and abstention thresholds.
4. Evaluate calibration, coverage/abstention, coefficient stability, runtime, and failure conditions—not only method-selection accuracy.
5. Keep the exact synthetic protocol and seed-level outputs with the paper artifacts; make no claim of universal superiority based on these first results.

## Leave-one-synthetic-family-out evaluation (25 October 2026)

A new runner, `research/run_leave_one_family_out.py`, was added and executed over five seeds (`20261025`–`20261029`), six held-out-family folds per seed, 12 training experiences for each of the other five families, and 30 test queries in the excluded family. This generated 900 held-out test queries over 30 folds. Each fold builds a fresh graph; the held-out family is not used to seed the memory for that fold.

Command:

```powershell
python -m research.run_leave_one_family_out --seeds 20261025 20261026 20261027 20261028 20261029 --train-per-family 12 --test-per-family 30 --output outputs/research_evaluation/leave_one_family_out
```

### Results observed

| Method | Mean accuracy across seeds | SD across seeds | Range |
|---|---:|---:|---:|
| Fixed Pearson | 50.0% | 0.0 pp | 50.0–50.0% |
| Fixed Spearman | 50.0% | 0.0 pp | 50.0–50.0% |
| Skew heuristic | 80.8% | 0.8 pp | 80.0–81.7% |
| Dataset-scoped history + heuristic fallback | 80.8% | 0.8 pp | 80.0–81.7% |
| Global-majority diagnostic + tie fallback | 0.0% | 0.0 pp | 0.0–0.0% |
| Contextual memory | 84.0% | 1.7 pp | 81.1–85.6% |

Contextual memory averaged 3.2 percentage points above the skew heuristic, but the family-specific results were mixed. It scored 90.0% on held-out clean-linear cases versus 100.0% for the heuristic, and 42.0% on the outlier-contaminated monotonic family versus 24.7% for the heuristic. Contextual recommendation coverage was 100% in this protocol, which motivates a dedicated out-of-distribution abstention test: the similarity threshold did not cause any test case to fall back in this benchmark.

### Important interpretation limits

This is transfer to one of six defined synthetic scenario families, not validation on independent real datasets. Training histories use oracle-labelled reference methods from the remaining families. The folds share overlapping training families, so across-seed confidence intervals are descriptive rather than independent-fold statistical evidence. The task-majority baseline's 0% result is a design artifact: because the six scenario families are balanced 3-vs-3 by reference method, excluding a single family leaves a 2-vs-3 label count and causes the majority baseline to always predict the opposite class for the held-out family. Treat that baseline as a demonstration of the weakness of context-free majority under this specific split protocol, not as a credible general baseline.

Artifacts are under `outputs/research_evaluation/leave_one_family_out/`. Next required experiments: real-dataset evaluation with independently specified quality criteria, out-of-distribution abstention calibration, and ablations of profile features, duplicate grouping, similarity thresholds, and `top_k`.


### Benchmark baseline clarification (25 October 2026)

The leave-one-family-out runner now reports two separate baselines. `dataset_history_fallback` calls the legacy task-and-dataset recommender using the held-out test dataset identifier; no exact-dataset history is available, so it falls back to the skew heuristic (80.8%, identical by design to that heuristic). `global_majority_heuristic_fallback` is a separate naive diagnostic that ignores dataset identity and contextual profiles. Its 0.0% LOFO result is caused by the six-family reference-label construction: after holding out one of three families of a class, the remaining five families contain a 2-vs-3 label count and the global majority always chooses the other class. This is a benchmark-design artifact, not a fair estimate of the performance of dataset-scoped historical memory. See `outputs/research_evaluation/leave_one_family_out/leave_one_family_out_report.md` for the corrected table and per-case files.

## Real-data perturbation pilot (25 October 2026)

The runner `research/run_real_dataset_robustness.py` now performs leave-one-dataset-out evaluation on four attributable datasets: scikit-learn Diabetes, UCI Wine Recognition, UCI Breast Cancer Wisconsin (Diagnostic), and UCI Iris. The project Titanic sample is excluded by default because the repository does not document its upstream provenance. The script constructs stability-derived strategy labels on training datasets only, then queries held-out data contexts after MCAR and extreme-outlier perturbations.

Run the reproduced pilot with:

```powershell
python -m research.run_real_dataset_robustness --seeds 20261025 20261026 20261027 --max-pairs-per-dataset 20 --perturbation-replicates 2 --output outputs/research_evaluation/real_dataset_robustness
```

The run evaluated 66 distinct feature pairs and 792 perturbed queries. Mean dataset-macro agreement with the empirical stability target was 84.6% for fixed Spearman, 84.6% for task-majority fallback, 83.3% for contextual memory, 34.0% for the skew heuristic, and 15.4% for fixed Pearson. Contextual memory's mean coefficient shift (0.0319) was nearly the same as fixed Spearman/task voting (0.0318). On the MCAR subset, contextual memory achieved 49.3% compared with 53.7% for fixed Spearman; on the outlier subset, contextual memory tied with fixed Spearman at 99.1%. The system recommended a strategy on every query, so out-of-distribution abstention was not tested.

Interpretation: this pilot does not demonstrate a real-data advantage for contextual memory over the simpler fixed-Spearman baseline under the chosen perturbation-stability objective. It is a controlled robustness experiment on real observations, not evidence of scientific truth or universal method quality. The next research task is to improve the objective-specific contextual policy and calibrate abstention, then broaden the evaluation with additional independently licensed datasets, perturbation protocols, and pair-clustered uncertainty estimates. Do not claim general superiority based on this pilot.
