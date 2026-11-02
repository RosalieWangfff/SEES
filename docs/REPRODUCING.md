# Reproduction protocol and commands

All commands below run from the repository root on Windows with prepared assets
under `runtime`. They were packaged from existing experiment entry points;
full training has not been rerun as part of release preparation.
Do not run simultaneous trainers against the same output directory.

## Fixed protocol

Datasets: MELD, DailyDialog, EmoryNLP. Seeds: 1008, 42, 43.
Selection: validation Weighted-F1, earliest tie; Macro-F1 from the same checkpoint.
Scores are percentages. Mean and sample SD are computed across the declared seeds.
MELD/EmoryNLP budget: 5 epochs. DailyDialog: 16 epochs.
The existing implementation records differences in initialization, feature pooling,
precision and batch sizes. These differences must remain visible in reporting.

## Main configuration and matched control

Use the four `no_evidence_control.py` commands in the README. This trains both
full typed SEES and its no-evidence control, generating new results in
`runtime/results_evidence_control_v2/` and `runtime/test_results_evidence_control_v2/`.
These new runs belong to the matched-control collection, not the archived main table.

## GLP-text, word/utterance crossing, relation shuffle

```cmd
python scripts/prepare.py --root runtime --install
python scripts/run.py --root runtime --stage glp --execute
python scripts/run.py --root runtime --stage cross --execute
python scripts/run.py --root runtime --stage relations --execute
python scripts/run.py --root runtime --stage evaluate --manifest runtime/nc_revision_reports/glp_manifest.json --execute
python scripts/run.py --root runtime --stage evaluate --manifest runtime/nc_revision_reports/cross_manifest.json --execute
python scripts/run.py --root runtime --stage evaluate --manifest runtime/nc_revision_reports/relations_manifest.json --execute
```

Without `--execute`, the runner plans missing training rather than launching it.
The `cross` stage covers G1 and G_word; neither removes the whole word evidence
path. The relation shuffle preserves connectivity and permutes relation labels
within each utterance. It is not a topology-removal experiment.

## DialogInfer-S-adapted

Example for one dataset and seed:

```cmd
python scripts/dialoginfer_s.py --root runtime --dataset meld --seed 1008 --mode train
python scripts/dialoginfer_s.py --root runtime --dataset meld --seed 1008 --mode test
```

Repeat for all three datasets and seeds. This implementation loads upstream
DialogueInfer classes using the project's aligned feature protocol. It takes
history text embeddings and speaker identity, without SEES's historical-emotion
input branch. Equal prediction tasks do not imply identical input information.

## Archived results

Legacy main-table and matched-control checkpoints have distinct identities.
Never edit a recorded checkpoint/protocol hash merely to bypass a mismatch.
The `recovery/` utilities retain historical compatibility logic; the recommended
new matched-control workflow is the v2 entry point above.

`runtime/evidence/recovered_base_manifest.json` is intentionally empty: this release
does not ship old recovered checkpoints. Add only records paired with verified
weights if explicitly restoring historical runs.

Some auxiliary producers retained for source hashing contain historical absolute
paths (`formal_model/english_predicted/train.py`, `erc_cache.py`,
`train_extension_extractor.py`, and the CPED branch in `extension_extractor_data.py`).
Do not invoke those legacy producers directly as portable commands. The documented
`nc_revision` and `evidence_control_v2` launchers locate their project root from
their installed files. Preserving these source bytes avoids invalidating archived
hashes; a future raw-data release must make the producer paths configurable and
record a new source version.
