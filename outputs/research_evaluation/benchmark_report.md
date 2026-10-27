# Synthetic Correlation Strategy Benchmark

> **Scope:** This is a synthetic proof-of-concept benchmark. It is not a claim of real-world superiority.

## Protocol

- Random seed: `20261025`
- Oracle-labelled training experiences: 72
- Held-out synthetic test cases: 180
- Scenario families: 6
- Training label counts: `{'pearson': 36, 'spearman': 36}`
- Train/test use disjoint random seeds. Scenario families are represented in both splits; this is held-out-sample testing, not unseen-family transfer.
- Each family randomizes sample size by approximately +/-15% around its base size. Noise scale varies by seed, and missingness/outlier contamination varies where applicable.
- For each family, the reference method encodes the declared analytical objective. Pearson and Spearman are not treated as universally interchangeable or universally optimal.
- Task-level majority voting ignores profile features. A tied majority abstains and falls back to the skew-based heuristic; this is labelled `task_majority_fallback`.
- Contextual memory is the project's actual `SelfEvolutionEngine` recommender using synthetic oracle-labelled prior experiences.

## Aggregate results

| Method | Selection accuracy | Mean absolute coefficient error | Historical coverage | Fallback cases |
|---|---:|---:|---:|---:|
| fixed_pearson | 50.0% (90/180) | 0.1363 | N/A | N/A |
| fixed_spearman | 50.0% (90/180) | 0.0699 | N/A | N/A |
| skew_heuristic | 81.7% (147/180) | 0.1258 | N/A | N/A |
| task_majority_fallback | 81.7% (147/180) | 0.1258 | 0.0% | 180 |
| contextual_memory | 97.8% (176/180) | 0.0690 | 100.0% | 0 |

## Per-scenario selection accuracy

| Scenario | Reference method | Pearson | Spearman | Skew heuristic | Task-majority fallback | Contextual memory | Contextual coverage |
|---|---|---:|---:|---:|---:|---:|---:|
| linear_clean | pearson | 100.0% | 0.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| linear_missing | pearson | 100.0% | 0.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| linear_small_sample | pearson | 100.0% | 0.0% | 93.3% | 93.3% | 100.0% | 100.0% |
| monotonic_exponential | spearman | 0.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| monotonic_cubic | spearman | 0.0% | 100.0% | 73.3% | 73.3% | 86.7% | 100.0% |
| monotonic_with_outliers | spearman | 0.0% | 100.0% | 23.3% | 23.3% | 100.0% | 100.0% |

## Paired, stratified bootstrap comparisons

The confidence intervals below are for the absolute difference in selection accuracy, with test cases resampled within scenario families.

| Comparison | Mean accuracy difference | 95% bootstrap interval |
|---|---:|---:|
| contextual_vs_heuristic | +16.2% | [+12.8%, +20.0%] |
| contextual_vs_task_majority_fallback | +16.1% | [+12.2%, +19.4%] |

## Limitations and interpretation

Reference methods are specified per scenario as a benchmark protocol, not universal truths. Training experiences use oracle-labelled synthetic scenarios. The experiment does not establish real-world generalization or statistical superiority; validate separately on held-out real datasets and quality-based outcomes.

The training memory is seeded with scenario-reference labels. This gives the contextual retrieval algorithm labelled prior experiences; it does not test whether the application's current structural-verification success flag independently identifies statistically high-quality strategies.

Coefficient error is calculated against a large, uncontaminated reference population for the selected estimator. It supplements strategy-selection accuracy and should not be interpreted as a universal quality ordering between Pearson and Spearman, since they estimate different association quantities.

The next experiment should repeat this protocol across random seeds, include additional scenario families and held-out licensed real datasets, and validate labels through an objective-specific evaluation protocol.

Runtime: 3.473 seconds. Versions: Python 3.13.5, NumPy 2.3.5, pandas 2.2.3, SciPy 1.17.0.


## Duplicate-history stress test

The stress-test corpus starts with strategy counts `{'pearson': 36, 'spearman': 36}`. It then adds 20 copies of one existing Pearson experience with the exact same dataset, column pair, and context.
Raw task-level majority changes from `tie / abstain` to `pearson`.
Contextual recommendation before replay: `Spearman correlation`; after replay: `Spearman correlation`.
Contextual recommendation and support unchanged: `True`.

This stress test illustrates duplicate suppression under a deliberately repeated history entry; it is not evidence about real-world performance and the replay copies are not independent observations.
