# Leave-One-Synthetic-Family-Out Results

- Seeds: `[20261025, 20261026, 20261027, 20261028, 20261029]`
- Held-out scenario families: 6
- Total held-out test cases: 900
- Fold runs: 30
- Elapsed seconds: 15.925

## Overall selection accuracy

| Method | Mean across seeds | SD | 95% t interval across seeds | Range |
|---|---:|---:|---:|---:|
| fixed_pearson | 50.0% | 0.0% | [50.0%, 50.0%] | [50.0%, 50.0%] |
| fixed_spearman | 50.0% | 0.0% | [50.0%, 50.0%] | [50.0%, 50.0%] |
| skew_heuristic | 80.8% | 0.8% | [79.7%, 81.8%] | [80.0%, 81.7%] |
| dataset_history_fallback | 80.8% | 0.8% | [79.7%, 81.8%] | [80.0%, 81.7%] |
| global_majority_heuristic_fallback | 0.0% | 0.0% | [0.0%, 0.0%] | [0.0%, 0.0%] |
| contextual_memory | 84.0% | 1.7% | [81.9%, 86.1%] | [81.1%, 85.6%] |

## Performance for each excluded family

| Held-out family | Reference label | Heuristic | Dataset-history fallback | Global-majority fallback | Contextual memory | Contextual coverage |
|---|---|---:|---:|---:|---:|---:|
| linear_clean | pearson | 100.0% | 100.0% | 0.0% | 90.0% | 100.0% |
| linear_missing | pearson | 100.0% | 100.0% | 0.0% | 100.0% | 100.0% |
| linear_small_sample | pearson | 100.0% | 100.0% | 0.0% | 100.0% | 100.0% |
| monotonic_exponential | spearman | 100.0% | 100.0% | 0.0% | 100.0% | 100.0% |
| monotonic_cubic | spearman | 60.0% | 60.0% | 0.0% | 72.0% | 100.0% |
| monotonic_with_outliers | spearman | 24.7% | 24.7% | 0.0% | 42.0% | 100.0% |

## Protocol and limitations

Each fold excludes the complete held-out family from the memory. Test cases are generated with separate seeds. Historical memory remains seeded with oracle-labelled reference methods for the non-held-out synthetic families. The dataset-scoped historical baseline receives a previously unseen test dataset name, so it has no eligible exact-dataset history and falls back to the skew heuristic. A separate global-majority-with-heuristic-fallback diagnostic intentionally ignores dataset identity and contextual profiles.

This tests transfer to an unseen family within six synthetic scenario families, not generalization to real datasets. Reference labels are defined by the benchmark protocol and are not universal statistical rules. Because the six families are balanced 3-vs-3 by reference method, excluding any one family leaves a 2-vs-3 family count; the global-majority diagnostic consequently chooses the opposite reference class in this designed protocol. This is a benchmark-design effect, not evidence that all task-level memory systems would achieve zero accuracy.

The dataset-history fallback uses exact dataset isolation and therefore falls back to the data-based heuristic for new held-out dataset IDs. The global-majority baseline ignores dataset identity; its 0% accuracy is induced by the 3-vs-3 family label design and must not be interpreted as a realistic universal baseline. The confidence intervals summarize across-seed variation descriptively. The number of seeds is small, and held-out folds reuse overlapping training families; do not interpret these intervals as definitive evidence of superiority.
