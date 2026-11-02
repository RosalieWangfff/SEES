# Model names and experimental controls

The main configuration follows the supplied 2026-11-02 manuscript: SEES is
G2 with typed dependencies on every dataset. Earlier working names SEES-W and
SEES-Neg are replaced in this release's documentation by explicit configuration
names. Existing code identifiers remain unchanged.

| Paper configuration | Code identifier | What changes |
| --- | --- | --- |
| SEES / G2 + typed | `G2__dep_typed` | History conditions word and utterance attention; typed dependency encoding |
| G1 + typed | `G1__dep_typed` | Neither attention level is conditioned on history; final history branch retained |
| G_word + typed | `G_word__dep_typed` | Only word attention is conditioned on history |
| G2, no structure | `G2` | Attention retained; structural enhancement removed |
| G1 / G_word, no structure | `G1` / `G_word` | Corresponding conditioning choice without structural enhancement |
| G2, untyped dependency | `G2__dep_untyped` | Connections/directions retained; generic direction embeddings |
| G2, shuffled labels | `G2__shuffle_rel` | Relation labels permuted within utterances; connections/directions/counts retained |
| G2, local negation | `G2__neg_local` | Messages restricted to identified negation-related dependency edges |
| G2, typed + local negation | `G2__dep_neg_joint` | Joint structural variant |
| w/o evidence pathway | `G2__no_evidence` | Aggregated evidence set to zero in separately matched retraining |
| GLP-text | `text` (GLP runner) | Text modeling component based on DialogueGLP |
| DialogInfer-S | `DialogInfer-S-adapted` (result records) | Adapted sequence model via DialogueInferModel |

G1 is not a no-evidence model: it retains ordinary word-level and utterance-level
attention. All G1/G_word/G2 configurations retain the historical-emotion input
to the final prediction. The no-evidence control retains dialogue context,
role interaction, and the historical-state projection and output bias:
`delta_z_e = W_e [0; p] + b_e`. Allocated parameter counts match, but active
computation does not necessarily match.

The task predicts the next speaker's future emotion from observed history and
the known next-speaker identity. Future utterance text and future emotion labels
are excluded from the prediction inputs. Predicted ERC labels are used for the
main experiments; the archived gold-history comparison is auxiliary validation
analysis with differing batch sizes, not a matched test-set comparison.

The supplied manuscript reports configuration selection by development
weighted-F1 averaged over seeds and then equally over datasets. G2 + typed has
the highest reported aggregate (45.4603). This documents the reported selection
criterion; it does not independently establish when that criterion was fixed.
