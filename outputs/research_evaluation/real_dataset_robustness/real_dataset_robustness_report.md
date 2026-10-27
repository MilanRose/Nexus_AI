# Cross-Dataset Real-Data Perturbation Robustness Evaluation

## Scope and metric

This benchmark uses real tabular observations and applies controlled MCAR-missingness and extreme-outlier perturbations. Each feature pair has two clean-data reference coefficients (one Pearson, one Spearman). For each perturbation family, the reference strategy is whichever method has the lower mean absolute change from its own clean-data coefficient across the perturbation replicates. Cases where the difference is below the configured label margin are marked ambiguous and excluded from selection-accuracy scoring, but remain in coefficient-shift summaries.

This is a measure of perturbation stability under a declared objective, not a measure of scientific truth, causal correctness, or universal superiority of one correlation measure.

## Methods compared

- `fixed_pearson`: always Pearson.
- `fixed_spearman`: always Spearman.
- `skew_heuristic`: the project's current skew-based heuristic.
- `task_majority_fallback`: most common robust strategy label among training-dataset experiences; falls back to the skew heuristic when the training vote is tied or absent.
- `contextual_memory`: graph-based similarity retrieval from other datasets; uses the production heuristic fallback when no context recommendation is made.

## Cross-dataset protocol

- Dataset folds: 4 (one complete dataset held out per fold).
- Distinct numeric feature pairs evaluated per run: 66
- Seeds: 20261025, 20261026, 20261027
- Perturbation families: mcar_10pct, extreme_outliers_5pct
- Perturbation replicates per feature-pair/family: 2
- Label margin: 0.0050 absolute coefficient shift.
- For every fold, memory is populated only with feature pairs from datasets other than the held-out dataset. The held-out dataset's pairs are never inserted as training experiences.

## Dataset sources and attribution

- **diabetes** (442 rows; 10 numeric features; 20 pairs): scikit-learn load_diabetes dataset documentation. [https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html) 442 rows; 10 features; bundled by scikit-learn; documentation notes that some feature meanings are not fully explicit.
- **wine_recognition** (178 rows; 13 numeric features; 20 pairs): UCI Wine Recognition dataset; Aeberhard and Forina (1992). [https://doi.org/10.24432/C5PC7J](https://doi.org/10.24432/C5PC7J) 178 rows; 13 numeric features; class label excluded from correlation pairs; CC BY 4.0 according to UCI attribution records.
- **breast_cancer_diagnostic** (569 rows; 30 numeric features; 20 pairs): UCI Breast Cancer Wisconsin (Diagnostic); Wolberg, Mangasarian, Street, and Street (1993). [https://doi.org/10.24432/C5DW2B](https://doi.org/10.24432/C5DW2B) 569 rows; 30 numeric features; diagnosis label excluded; UCI states CC BY 4.0.
- **iris** (150 rows; 4 numeric features; 6 pairs): UCI Iris dataset; Fisher (1936). [https://doi.org/10.24432/C56C76](https://doi.org/10.24432/C56C76) 150 rows; 4 numeric measurements; species label excluded; UCI states CC BY 4.0.

## Aggregate results

| Method | Mean macro accuracy | SD across seeds | Seed range | Mean macro absolute coefficient shift |
|---|---:|---:|---:|---:|
| fixed_pearson | 0.1536 | 0.0596 | 0.0913–0.2100 | 0.0950 |
| fixed_spearman | 0.8464 | 0.0596 | 0.7900–0.9087 | 0.0318 |
| skew_heuristic | 0.3398 | 0.0441 | 0.2988–0.3864 | 0.0787 |
| task_majority_fallback | 0.8464 | 0.0596 | 0.7900–0.9087 | 0.0318 |
| contextual_memory | 0.8335 | 0.0534 | 0.7976–0.8948 | 0.0319 |

### Contextual-memory coverage

Mean decisive-label coverage: 70.1% of test cases.
Mean contextual recommendation coverage: 100.0% of queries.
The contextual system did not abstain on any query in this run, despite supporting fallback. Therefore this experiment does not validate out-of-distribution abstention or calibration.

## Results by perturbation family

| Perturbation | Method | Mean macro accuracy | SD across seeds | Mean macro absolute shift |
|---|---|---:|---:|---:|
| mcar_10pct | fixed_pearson | 0.4630 | 0.1334 | 0.0188 |
| mcar_10pct | fixed_spearman | 0.5370 | 0.1334 | 0.0188 |
| mcar_10pct | skew_heuristic | 0.4633 | 0.1017 | 0.0185 |
| mcar_10pct | task_majority_fallback | 0.5370 | 0.1334 | 0.0188 |
| mcar_10pct | contextual_memory | 0.4931 | 0.1108 | 0.0189 |
| extreme_outliers_5pct | fixed_pearson | 0.0088 | 0.0152 | 0.1712 |
| extreme_outliers_5pct | fixed_spearman | 0.9912 | 0.0152 | 0.0449 |
| extreme_outliers_5pct | skew_heuristic | 0.2902 | 0.0208 | 0.1390 |
| extreme_outliers_5pct | task_majority_fallback | 0.9912 | 0.0152 | 0.0449 |
| extreme_outliers_5pct | contextual_memory | 0.9912 | 0.0152 | 0.0449 |

## Per-dataset results

| Held-out dataset | Method | Mean accuracy | Mean absolute shift | Decisive cases / queries | Contextual recommendation coverage |
|---|---|---:|---:|---:|---:|
| diabetes | fixed_pearson | 0.1481 | 0.0741 | 54/80 | 100.0% |
| diabetes | fixed_spearman | 0.8519 | 0.0257 | 54/80 | 100.0% |
| diabetes | skew_heuristic | 0.1667 | 0.0735 | 54/80 | 100.0% |
| diabetes | task_majority_fallback | 0.8519 | 0.0257 | 54/80 | 100.0% |
| diabetes | contextual_memory | 0.8333 | 0.0256 | 54/80 | 100.0% |
| wine_recognition | fixed_pearson | 0.1724 | 0.0884 | 58/80 | 100.0% |
| wine_recognition | fixed_spearman | 0.8276 | 0.0349 | 58/80 | 100.0% |
| wine_recognition | skew_heuristic | 0.3621 | 0.0783 | 58/80 | 100.0% |
| wine_recognition | task_majority_fallback | 0.8276 | 0.0349 | 58/80 | 100.0% |
| wine_recognition | contextual_memory | 0.8276 | 0.0347 | 58/80 | 100.0% |
| breast_cancer_diagnostic | fixed_pearson | 0.2069 | 0.0811 | 58/80 | 100.0% |
| breast_cancer_diagnostic | fixed_spearman | 0.7931 | 0.0260 | 58/80 | 100.0% |
| breast_cancer_diagnostic | skew_heuristic | 0.5862 | 0.0415 | 58/80 | 100.0% |
| breast_cancer_diagnostic | task_majority_fallback | 0.7931 | 0.0260 | 58/80 | 100.0% |
| breast_cancer_diagnostic | contextual_memory | 0.7931 | 0.0260 | 58/80 | 100.0% |
| iris | fixed_pearson | 0.1111 | 0.1340 | 18/24 | 100.0% |
| iris | fixed_spearman | 0.8889 | 0.0420 | 18/24 | 100.0% |
| iris | skew_heuristic | 0.2222 | 0.1296 | 18/24 | 100.0% |
| iris | task_majority_fallback | 0.8889 | 0.0420 | 18/24 | 100.0% |
| iris | contextual_memory | 0.7778 | 0.0427 | 18/24 | 100.0% |
| diabetes | fixed_pearson | 0.1111 | 0.0804 | 54/80 | 100.0% |
| diabetes | fixed_spearman | 0.8889 | 0.0290 | 54/80 | 100.0% |
| diabetes | skew_heuristic | 0.1111 | 0.0804 | 54/80 | 100.0% |
| diabetes | task_majority_fallback | 0.8889 | 0.0290 | 54/80 | 100.0% |
| diabetes | contextual_memory | 0.8889 | 0.0291 | 54/80 | 100.0% |
| wine_recognition | fixed_pearson | 0.1111 | 0.0874 | 54/80 | 100.0% |
| wine_recognition | fixed_spearman | 0.8889 | 0.0297 | 54/80 | 100.0% |
| wine_recognition | skew_heuristic | 0.3519 | 0.0719 | 54/80 | 100.0% |
| wine_recognition | task_majority_fallback | 0.8889 | 0.0297 | 54/80 | 100.0% |
| wine_recognition | contextual_memory | 0.8333 | 0.0305 | 54/80 | 100.0% |
| breast_cancer_diagnostic | fixed_pearson | 0.1429 | 0.0918 | 56/80 | 100.0% |
| breast_cancer_diagnostic | fixed_spearman | 0.8571 | 0.0293 | 56/80 | 100.0% |
| breast_cancer_diagnostic | skew_heuristic | 0.6607 | 0.0457 | 56/80 | 100.0% |
| breast_cancer_diagnostic | task_majority_fallback | 0.8571 | 0.0293 | 56/80 | 100.0% |
| breast_cancer_diagnostic | contextual_memory | 0.8571 | 0.0293 | 56/80 | 100.0% |
| iris | fixed_pearson | 0.0000 | 0.1276 | 14/24 | 100.0% |
| iris | fixed_spearman | 1.0000 | 0.0445 | 14/24 | 100.0% |
| iris | skew_heuristic | 0.0714 | 0.1222 | 14/24 | 100.0% |
| iris | task_majority_fallback | 1.0000 | 0.0445 | 14/24 | 100.0% |
| iris | contextual_memory | 1.0000 | 0.0445 | 14/24 | 100.0% |
| diabetes | fixed_pearson | 0.1200 | 0.0762 | 50/80 | 100.0% |
| diabetes | fixed_spearman | 0.8800 | 0.0255 | 50/80 | 100.0% |
| diabetes | skew_heuristic | 0.1600 | 0.0711 | 50/80 | 100.0% |
| diabetes | task_majority_fallback | 0.8800 | 0.0255 | 50/80 | 100.0% |
| diabetes | contextual_memory | 0.8800 | 0.0254 | 50/80 | 100.0% |
| wine_recognition | fixed_pearson | 0.3103 | 0.0828 | 58/80 | 100.0% |
| wine_recognition | fixed_spearman | 0.6897 | 0.0322 | 58/80 | 100.0% |
| wine_recognition | skew_heuristic | 0.4828 | 0.0679 | 58/80 | 100.0% |
| wine_recognition | task_majority_fallback | 0.6897 | 0.0322 | 58/80 | 100.0% |
| wine_recognition | contextual_memory | 0.7759 | 0.0317 | 58/80 | 100.0% |
| breast_cancer_diagnostic | fixed_pearson | 0.1875 | 0.0847 | 64/80 | 100.0% |
| breast_cancer_diagnostic | fixed_spearman | 0.8125 | 0.0256 | 64/80 | 100.0% |
| breast_cancer_diagnostic | skew_heuristic | 0.6250 | 0.0405 | 64/80 | 100.0% |
| breast_cancer_diagnostic | task_majority_fallback | 0.8125 | 0.0256 | 64/80 | 100.0% |
| breast_cancer_diagnostic | contextual_memory | 0.8125 | 0.0256 | 64/80 | 100.0% |
| iris | fixed_pearson | 0.2222 | 0.1313 | 18/24 | 100.0% |
| iris | fixed_spearman | 0.7778 | 0.0373 | 18/24 | 100.0% |
| iris | skew_heuristic | 0.2778 | 0.1224 | 18/24 | 100.0% |
| iris | task_majority_fallback | 0.7778 | 0.0373 | 18/24 | 100.0% |
| iris | contextual_memory | 0.7222 | 0.0373 | 18/24 | 100.0% |

## Limitations and interpretation

- The label is the method with lower mean absolute coefficient shift relative to its own clean-data estimate under a declared perturbation family; it is not scientific ground truth.
- Perturbations are controlled stress tests applied to real observations, not naturally occurring distribution shifts.
- Datasets differ in domain and size; a small number of datasets limits domain-level inference.
- Contextual memory is trained only on pairs from datasets other than the held-out dataset in each fold.
- The Titanic sample's upstream provenance is not documented in this repository and should be verified before publication.
- The target method is computed from each held-out pair's perturbation stability and is used only for evaluation; it is not supplied to the held-out memory.
- Correlation coefficients are not interchangeable: Pearson measures linear association, while Spearman measures rank-based monotonic association. Stability relative to each method's own clean estimate is a deliberately narrow robustness objective.
- The held-out unit is an entire dataset, which reduces direct row/pair leakage. The number of datasets is still small, so results are descriptive rather than a definitive population-level inference.
- The default run excludes the project's Titanic sample because its upstream provenance is undocumented; use `--include-titanic-sample` only for exploratory analysis until provenance is verified.
- The benchmark should be complemented by domain-specific analysis goals and additional independent real datasets before a conference-paper claim is made.

## Reproducibility metadata

- Python: 3.13.5
- pandas: 2.2.3
- NumPy: 2.3.5
- SciPy: 1.17.0
- scikit-learn: 1.8.0
- Runtime: 27.18 seconds
