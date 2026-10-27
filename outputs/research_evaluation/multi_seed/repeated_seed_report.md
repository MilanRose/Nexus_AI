# Repeated-Seed Synthetic Benchmark Summary

- Independent random seeds: `[20261025, 20261026, 20261027, 20261028, 20261029]`
- Training experiences per seed: 72
- Test cases per seed: 180
- Total held-out synthetic evaluations: 900
- The interval is a two-sided 95% Student-t interval across run-level accuracies; it is descriptive and based on the number of seeds, not the individual test rows.

## Results

| Method | Mean accuracy | SD across seeds | 95% t interval | Range | Mean coefficient error |
|---|---:|---:|---:|---:|---:|
| fixed_pearson | 50.0% | 0.0% | [50.0%, 50.0%] | [50.0%, 50.0%] | 0.1492 |
| fixed_spearman | 50.0% | 0.0% | [50.0%, 50.0%] | [50.0%, 50.0%] | 0.0794 |
| skew_heuristic | 82.0% | 0.3% | [81.6%, 82.4%] | [81.7%, 82.2%] | 0.1389 |
| task_majority_fallback | 82.0% | 0.3% | [81.6%, 82.4%] | [81.7%, 82.2%] | 0.1389 |
| contextual_memory | 97.7% | 0.2% | [97.4%, 98.0%] | [97.2%, 97.8%] | 0.0772 |

## Per-scenario accuracy across seeds

| Scenario | Reference | Pearson | Spearman | Skew heuristic | Contextual memory | Contextual coverage |
|---|---|---:|---:|---:|---:|---:|
| linear_clean | pearson | 100.0% | 0.0% | 100.0% | 100.0% | 100.0% |
| linear_missing | pearson | 100.0% | 0.0% | 100.0% | 100.0% | 100.0% |
| linear_small_sample | pearson | 100.0% | 0.0% | 93.3% | 98.7% | 100.0% |
| monotonic_exponential | spearman | 0.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| monotonic_cubic | spearman | 0.0% | 100.0% | 76.0% | 87.3% | 100.0% |
| monotonic_with_outliers | spearman | 0.0% | 100.0% | 22.7% | 100.0% | 100.0% |

## Paired seed-level accuracy differences

| Comparison | Mean difference | 95% t interval |
|---|---:|---:|
| contextual_vs_heuristic | +15.7% | [+15.1%, +16.2%] |
| contextual_vs_task_majority_fallback | +15.7% | [+15.1%, +16.2%] |

## Interpretation limits

Means and t-based intervals describe variability across the specified random seeds under this synthetic benchmark protocol. They do not establish generalization to unseen scientific domains or real datasets. Training history uses oracle-labelled scenario experiences; repeat on held-out real datasets and independent quality labels.

The benchmark is based on a controlled synthetic scenario policy. The next required step before a conference performance claim is replication with held-out real datasets and quality labels validated independently of the strategy selector.
