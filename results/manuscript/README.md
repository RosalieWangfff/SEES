# Manuscript result tables

Numeric exports from the author-supplied 2026-11-02 bilingual review manuscript.
F1 values use a 0–100 scale. Summary tables report means and sample SD over
three seeds except the explicitly marked archived history-source validation.
Preserve the distinction between archived main results and matched retraining.

| File | Contents |
| --- | --- |
| [datasets.csv](datasets.csv) | Prediction-instance counts |
| [main_results.csv](main_results.csv) | Main SEES versus baselines |
| [conditioning_typed.csv](conditioning_typed.csv) | Conditioning placement, typed structure fixed |
| [structure.csv](structure.csv) | G2: no structure versus typed |
| [relation_labels.csv](relation_labels.csv) | Untyped, typed, shuffled relation labels |
| [local_negation.csv](local_negation.csv) | Typed, local negation, joint |
| [evidence_path.csv](evidence_path.csv) | Separate matched evidence-path retraining |
| [history_source_validation.csv](history_source_validation.csv) | Auxiliary archived EmoryNLP validation; not matched |
| [development_selection.csv](development_selection.csv) | Reported development configuration selection |
| [paired_conditioning.csv](paired_conditioning.csv) | Within-seed G_word + typed minus G1 + typed |
| [conditioning_no_structure.csv](conditioning_no_structure.csv) | Appendix: conditioning without structure |
| [per_seed_main_and_ablations.csv](per_seed_main_and_ablations.csv) | 108 archived test runs, rounded to three decimals |
| [per_seed_evidence_path.csv](per_seed_evidence_path.csv) | 18 matched-control test runs, rounded to three decimals |
| [class_structure.csv](class_structure.csv) | Per-class F1: no structure versus typed |
| [class_conditioning.csv](class_conditioning.csv) | Per-class F1: typed G1, G_word, G2 |

Source tables use the draft's displayed precision. Full-precision audit
summaries remain in `../test_summary.csv`; matched runs remain in
`../no_evidence_control_runs.csv`. Class-level differences can differ from
subtraction of displayed values because the source uses unrounded values.

See [interpretation](../../docs/RESULTS.md) and [configuration mapping](../../docs/MODEL.md).
