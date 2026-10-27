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
- Distinct numeric feature pairs evaluated per run: 30
- Seeds: 20261025, 20261026, 20261027
- Perturbation families: mcar_10pct, extreme_outliers_5pct
- Perturbation replicates per feature-pair/family: 1
- Label margin: 0.0050 absolute coefficient shift.
- For every fold, memory is populated only with feature pairs from datasets other than the held-out dataset. The held-out dataset's pairs are never inserted as training experiences.

## Dataset sources and attribution

- **diabetes** (442 rows; 10 numeric features; 8 pairs): scikit-learn load_diabetes dataset documentation. [https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html) 442 rows; 10 features; bundled by scikit-learn; documentation notes that some feature meanings are not fully explicit.
- **wine_recognition** (178 rows; 13 numeric features; 8 pairs): UCI Wine Recognition dataset; Aeberhard and Forina (1992). [https://doi.org/10.24432/C5PC7J](https://doi.org/10.24432/C5PC7J) 178 rows; 13 numeric features; class label excluded from correlation pairs; CC BY 4.0 according to UCI attribution records.
- **breast_cancer_diagnostic** (569 rows; 30 numeric features; 8 pairs): UCI Breast Cancer Wisconsin (Diagnostic); Wolberg, Mangasarian, Street, and Street (1993). [https://doi.org/10.24432/C5DW2B](https://doi.org/10.24432/C5DW2B) 569 rows; 30 numeric features; diagnosis label excluded; UCI states CC BY 4.0.
- **iris** (150 rows; 4 numeric features; 6 pairs): UCI Iris dataset; Fisher (1936). [https://doi.org/10.24432/C56C76](https://doi.org/10.24432/C56C76) 150 rows; 4 numeric measurements; species label excluded; UCI states CC BY 4.0.

## Aggregate results

| Method | Mean macro accuracy | SD across seeds | Seed range | Mean macro absolute coefficient shift |
|---|---:|---:|---:|---:|
| fixed_pearson | 0.1790 | 0.0418 | 0.1389–0.2222 | 0.0954 |
| fixed_spearman | 0.8210 | 0.0418 | 0.7778–0.8611 | 0.0338 |
| skew_heuristic | 0.3335 | 0.0541 | 0.2986–0.3958 | 0.0808 |
| task_majority_fallback | 0.8210 | 0.0418 | 0.7778–0.8611 | 0.0338 |
| contextual_memory | 0.8303 | 0.0351 | 0.7986–0.8681 | 0.0335 |

### Contextual-memory coverage

Mean decisive-label coverage: 72.4% of test cases.
Mean contextual recommendation coverage: 100.0% of queries.
The contextual system did not abstain on any query in this run, despite supporting fallback. Therefore this experiment does not validate out-of-distribution abstention or calibration.

## Results by perturbation family

| Perturbation | Method | Mean macro accuracy | SD across seeds | Mean macro absolute shift |
|---|---|---:|---:|---:|
| mcar_10pct | fixed_pearson | 0.5028 | 0.1156 | 0.0213 |
| mcar_10pct | fixed_spearman | 0.4972 | 0.1156 | 0.0209 |
| mcar_10pct | skew_heuristic | 0.5028 | 0.1071 | 0.0203 |
| mcar_10pct | task_majority_fallback | 0.4972 | 0.1156 | 0.0209 |
| mcar_10pct | contextual_memory | 0.5000 | 0.1201 | 0.0204 |
| extreme_outliers_5pct | fixed_pearson | 0.0104 | 0.0180 | 0.1695 |
| extreme_outliers_5pct | fixed_spearman | 0.9896 | 0.0180 | 0.0467 |
| extreme_outliers_5pct | skew_heuristic | 0.2624 | 0.0166 | 0.1412 |
| extreme_outliers_5pct | task_majority_fallback | 0.9896 | 0.0180 | 0.0467 |
| extreme_outliers_5pct | contextual_memory | 0.9896 | 0.0180 | 0.0467 |

## Per-dataset results

| Held-out dataset | Method | Mean accuracy | Mean absolute shift | Decisive cases / queries | Contextual recommendation coverage |
|---|---|---:|---:|---:|---:|
| diabetes | fixed_pearson | 0.1111 | 0.0780 | 9/16 | 100.0% |
| diabetes | fixed_spearman | 0.8889 | 0.0266 | 9/16 | 100.0% |
| diabetes | skew_heuristic | 0.1111 | 0.0780 | 9/16 | 100.0% |
| diabetes | task_majority_fallback | 0.8889 | 0.0266 | 9/16 | 100.0% |
| diabetes | contextual_memory | 0.8889 | 0.0266 | 9/16 | 100.0% |
| wine_recognition | fixed_pearson | 0.2308 | 0.0843 | 13/16 | 100.0% |
| wine_recognition | fixed_spearman | 0.7692 | 0.0400 | 13/16 | 100.0% |
| wine_recognition | skew_heuristic | 0.3077 | 0.0810 | 13/16 | 100.0% |
| wine_recognition | task_majority_fallback | 0.7692 | 0.0400 | 13/16 | 100.0% |
| wine_recognition | contextual_memory | 0.7692 | 0.0400 | 13/16 | 100.0% |
| breast_cancer_diagnostic | fixed_pearson | 0.2500 | 0.0937 | 12/16 | 100.0% |
| breast_cancer_diagnostic | fixed_spearman | 0.7500 | 0.0280 | 12/16 | 100.0% |
| breast_cancer_diagnostic | skew_heuristic | 0.5833 | 0.0554 | 12/16 | 100.0% |
| breast_cancer_diagnostic | task_majority_fallback | 0.7500 | 0.0280 | 12/16 | 100.0% |
| breast_cancer_diagnostic | contextual_memory | 0.7500 | 0.0280 | 12/16 | 100.0% |
| iris | fixed_pearson | 0.1111 | 0.1331 | 9/12 | 100.0% |
| iris | fixed_spearman | 0.8889 | 0.0418 | 9/12 | 100.0% |
| iris | skew_heuristic | 0.2222 | 0.1240 | 9/12 | 100.0% |
| iris | task_majority_fallback | 0.8889 | 0.0418 | 9/12 | 100.0% |
| iris | contextual_memory | 0.8889 | 0.0418 | 9/12 | 100.0% |
| diabetes | fixed_pearson | 0.0833 | 0.0814 | 12/16 | 100.0% |
| diabetes | fixed_spearman | 0.9167 | 0.0249 | 12/16 | 100.0% |
| diabetes | skew_heuristic | 0.0833 | 0.0814 | 12/16 | 100.0% |
| diabetes | task_majority_fallback | 0.9167 | 0.0249 | 12/16 | 100.0% |
| diabetes | contextual_memory | 0.8333 | 0.0256 | 12/16 | 100.0% |
| wine_recognition | fixed_pearson | 0.3333 | 0.0636 | 12/16 | 100.0% |
| wine_recognition | fixed_spearman | 0.6667 | 0.0303 | 12/16 | 100.0% |
| wine_recognition | skew_heuristic | 0.5833 | 0.0472 | 12/16 | 100.0% |
| wine_recognition | task_majority_fallback | 0.6667 | 0.0303 | 12/16 | 100.0% |
| wine_recognition | contextual_memory | 0.9167 | 0.0279 | 12/16 | 100.0% |
| breast_cancer_diagnostic | fixed_pearson | 0.2500 | 0.0897 | 12/16 | 100.0% |
| breast_cancer_diagnostic | fixed_spearman | 0.7500 | 0.0323 | 12/16 | 100.0% |
| breast_cancer_diagnostic | skew_heuristic | 0.5833 | 0.0407 | 12/16 | 100.0% |
| breast_cancer_diagnostic | task_majority_fallback | 0.7500 | 0.0323 | 12/16 | 100.0% |
| breast_cancer_diagnostic | contextual_memory | 0.6667 | 0.0338 | 12/16 | 100.0% |
| iris | fixed_pearson | 0.2222 | 0.1210 | 9/12 | 100.0% |
| iris | fixed_spearman | 0.7778 | 0.0407 | 9/12 | 100.0% |
| iris | skew_heuristic | 0.3333 | 0.1103 | 9/12 | 100.0% |
| iris | task_majority_fallback | 0.7778 | 0.0407 | 9/12 | 100.0% |
| iris | contextual_memory | 0.7778 | 0.0407 | 9/12 | 100.0% |
| diabetes | fixed_pearson | 0.0833 | 0.0730 | 12/16 | 100.0% |
| diabetes | fixed_spearman | 0.9167 | 0.0267 | 12/16 | 100.0% |
| diabetes | skew_heuristic | 0.0833 | 0.0730 | 12/16 | 100.0% |
| diabetes | task_majority_fallback | 0.9167 | 0.0267 | 12/16 | 100.0% |
| diabetes | contextual_memory | 0.7500 | 0.0282 | 12/16 | 100.0% |
| wine_recognition | fixed_pearson | 0.0000 | 0.0867 | 9/16 | 100.0% |
| wine_recognition | fixed_spearman | 1.0000 | 0.0360 | 9/16 | 100.0% |
| wine_recognition | skew_heuristic | 0.2222 | 0.0775 | 9/16 | 100.0% |
| wine_recognition | task_majority_fallback | 1.0000 | 0.0360 | 9/16 | 100.0% |
| wine_recognition | contextual_memory | 1.0000 | 0.0361 | 9/16 | 100.0% |
| breast_cancer_diagnostic | fixed_pearson | 0.2500 | 0.0967 | 12/16 | 100.0% |
| breast_cancer_diagnostic | fixed_spearman | 0.7500 | 0.0335 | 12/16 | 100.0% |
| breast_cancer_diagnostic | skew_heuristic | 0.6667 | 0.0569 | 12/16 | 100.0% |
| breast_cancer_diagnostic | task_majority_fallback | 0.7500 | 0.0335 | 12/16 | 100.0% |
| breast_cancer_diagnostic | contextual_memory | 0.8333 | 0.0325 | 12/16 | 100.0% |
| iris | fixed_pearson | 0.2222 | 0.1440 | 9/12 | 100.0% |
| iris | fixed_spearman | 0.7778 | 0.0443 | 9/12 | 100.0% |
| iris | skew_heuristic | 0.2222 | 0.1440 | 9/12 | 100.0% |
| iris | task_majority_fallback | 0.7778 | 0.0443 | 9/12 | 100.0% |
| iris | contextual_memory | 0.8889 | 0.0413 | 9/12 | 100.0% |

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

- Python: 3.13.1
- pandas: 2.2.3
- NumPy: 2.2.4
- SciPy: 1.15.1
- scikit-learn: 1.6.1
- Runtime: 9.10 seconds
