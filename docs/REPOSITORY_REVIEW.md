# Reference repository review

Reviewed 2026-11-02; URLs and inspected commit identifiers are recorded in
`provenance/reference_repositories.json`.

| Repository | Observed organization | Applied here |
| --- | --- | --- |
| Reason-Wang/DialogueGLP | README with environment and feature extraction/training stages; model, data, scripts, utils; MIT license | Separate environment, assets and execution; retain upstream attribution |
| declare-lab/conv-emotion | Multiple named model directories, paper/citation information, architecture images and licensing | Explicit model names and framework figure; no claim that an adapted baseline is the exact original |
| multica-ai/andrej-karpathy-skills | Four principles: think before coding, simplicity, surgical changes, verifiable execution | Preserve audited scientific code; add thin release documentation and source verification |

Our additional needs are provenance-aware results, a clear distinction between
old and matched-retraining runs, and checkpoint availability. A source repository
alone does not make all archived results independently reproducible.
