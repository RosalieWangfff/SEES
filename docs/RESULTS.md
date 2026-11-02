# Results aligned with the 2026-11-02 manuscript

The publication-facing CSVs are indexed in [results/manuscript](../results/manuscript/README.md).
`test_summary.csv` retains the full-precision archived result summaries and source
labels. `no_evidence_control_runs.csv` retains 18 full-precision matched-control
runs. Exported manuscript values preserve the precision shown in the supplied
draft. Means and sample standard deviations always come from the same collection.

## Main results

| Dataset | Configuration | Weighted-F1 | Macro-F1 |
| --- | --- | ---: | ---: |
| MELD | GLP-text | 37.28 ± 0.53 | 19.53 ± 0.84 |
| MELD | DialogInfer-S | 37.13 ± 0.95 | 17.82 ± 0.64 |
| MELD | SEES | 38.04 ± 0.76 | 18.71 ± 0.51 |
| DailyDialog | GLP-text | 73.09 ± 1.06 | 30.03 ± 4.73 |
| DailyDialog | DialogInfer-S | 72.53 ± 1.05 | 31.04 ± 0.38 |
| DailyDialog | SEES | 73.74 ± 0.82 | 28.34 ± 0.48 |
| EmoryNLP | GLP-text | 20.15 ± 0.88 | 16.01 ± 0.93 |
| EmoryNLP | DialogInfer-S | 23.46 ± 1.27 | 19.99 ± 1.64 |
| EmoryNLP | SEES | 22.29 ± 0.56 | 18.62 ± 0.60 |

The main model is fixed G2 + typed. Its mean weighted-F1 exceeds the strongest
baseline by 0.76 points on MELD and 0.65 on DailyDialog. On EmoryNLP it exceeds
GLP-text but trails DialogInfer-S on both metrics. Macro-F1 trails GLP-text on
MELD and both baselines on DailyDialog. Report both metrics; do not describe
these results as uniformly better or as significant without a matching test.

## What the ablations support

- With typed structure fixed, G_word improves mean weighted-F1 and macro-F1 over
  G1 on all three datasets. Adding utterance-level conditioning (G2) gives mixed
  changes. Word-level conditioning has more consistent support in this setting.
- Without structure, G_word does not improve macro-F1 over G1 on any dataset.
  The conditioning result should therefore be scoped to the typed setting.
- Untyped dependency connectivity improves mean weighted-F1 over no structure
  on all three datasets. Typed labels do not consistently improve weighted-F1
  over untyped connections. Typed-versus-shuffled results also vary by metric.
- Local negation exceeds typed dependency in mean weighted-F1 on all three
  datasets, but macro-F1 improves only on MELD. It remains a structural variant,
  not a dataset-specific replacement for the fixed main configuration.
- The separately retrained full evidence path improves mean weighted-F1 and
  macro-F1 over no evidence on MELD and EmoryNLP, but slightly decreases both on
  DailyDialog. This supports a dataset-dependent incremental contribution.

Information overlap or irrelevant structural messages are possible explanations,
not mechanisms established by these score comparisons. Attention visualizations
illustrate model behavior; they do not prove causal importance.

## Separate matched evidence-path collection

| Dataset | Full W-F1 | No evidence W-F1 | Full Macro-F1 | No evidence Macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| MELD | 38.14 ± 0.79 | 37.69 ± 0.14 | 19.49 ± 0.63 | 18.93 ± 0.53 |
| DailyDialog | 73.72 ± 1.41 | 73.85 ± 0.58 | 28.51 ± 2.77 | 28.79 ± 2.67 |
| EmoryNLP | 21.66 ± 1.24 | 21.26 ± 0.76 | 17.45 ± 1.38 | 17.24 ± 0.89 |

Three seeds per row; these values are not the archived main-table full-model
values. The control zeros evidence while retaining the historical-emotion
projection and bias. Parameters can remain allocated but inactive.

## Auxiliary results and provenance

The gold-versus-predicted history comparison is an archived, single-seed EmoryNLP
validation comparison with batch sizes 24 and 32 respectively. It is not a
controlled estimate of the effect of history source. DailyDialog's seven-label
macro-F1 includes a zero-support fear class. Class-level tables give the support
counts and show where gains and losses occur.

MELD/EmoryNLP matched controls share a B_s42 historical-supervised warm start
within each dataset; DailyDialog uses fresh downstream initialization. The
three downstream seeds are not independent pretraining runs. The original
training-data provenance of archived warm-start and ERC checkpoints remains
unresolved. Downstream test records alone do not prove end-to-end data isolation.

`checkpoint_file_verified` identifies weights independently checked locally.
DailyDialog's returned matched results lack local checkpoint files; recorded
metadata hashes do not imply those files were rehashed here.

The table-alignment record in `provenance/manuscript_alignment.json` identifies
the supplied manuscript by SHA-256 and lists source lines and verification scope.
The case-study dialogue transcripts and full manuscript are not bundled into the
code release; this update exports numeric tables only.
