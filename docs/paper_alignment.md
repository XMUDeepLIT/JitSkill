# Paper-to-code mapping

The current paper uses the name **JitSkill**. This release follows the current Method and Method Details; it does not assert that newly corrected code reproduces existing experimental measurements.

| Paper component | Code | Contract |
|---|---|---|
| Explicit task progress, Eq. explicit-progress | `jitskill/explicit_record.py` | Evidence-only action/result record; goal and six recent actions/failures use appendix limits |
| Optional structured facts/subtask progress | `ExplicitRecord.observe` | Adapter-supplied values require direct evidence in the observation; no inferred completion |
| Latent update, Eq. latent-state | `jitskill/latent_state.py` | Frozen encoder base plus LoRA, initial state and 16 slots; preceding transition only |
| Truncated recurrent training | `episode_forward` | Forward state persists; gradients detach between updates |
| History-compression pretraining, Eq. memory-training | `jitskill/train_compression.py` | Pretrains encoder LoRA, slots, and recurrent projection before navigation fine-tuning |
| Navigation, Eq. nav-training | `jitskill/train_navigator.py`, `navigator.py` | Initializes from compression checkpoint; uniform labels over next-action supporting skills jointly optimize encoder and navigator |
| Query and transferable description scoring | `Navigator`, `CrossScorer` | Joint input of mean latent slots and frozen record embedding; description-conditioned cross head for new libraries |
| Skill selection | `jitskill/fusion.py` | Learned/current-step dense scores and BM25 fused by reciprocal ranks, procedural mask, top-k token packing |
| Mapping, Eq. latent-state-projection | `jitskill/executor/projection.py` | Linear P_psi maps d_z to the actual executor embedding dimension |
| Frozen executor likelihood, Eq. executor-training | `train_projection.py` | Only P_psi is optimized; no executor LoRA, merge, or executor parameter updates |
| History reconstruction / JSD / contrast / weighting | `jitskill/objectives.py`, `train_compression.py` | Encoder pretraining objective before navigation fine-tuning |
| Skill injection and shared state | `serving/inject_proxy.py` | Same B_t and Z_t used for navigation and execution; only Z_t is projected for executor |
| Prompt templates | `prompts/` | Archived appendix template plus a proxy-compatible JitSkill runtime template; both use the paper's injected-block text |

## Corrections relative to the supplied package

- Replace executor LoRA adaptation and direct latent insertion with projection-only training and projected state tokens for a frozen executor.
- Prevent target-action history leakage; align before-decision transitions in training and serving.
- Retain action outcomes and failures in a bounded explicit record, rather than just API names. Embed and inject the same record.
- Use the appendix's reference-only skill header. Support the paper's user/tool/system injection locations.
- Do not inject masked declarative cards. Search the full ranked procedural pool when earlier skills do not fit the budget.
- Fold unseen history transitions once per episode; repeated requests do not duplicate the recurrence.
- Remove internal experiment recipes, deployment paths, and executor-merging scripts.

## Paper ambiguities and release choices

- The current Method figure says six latent slots and shows compression objectives without the newer projection/navigation details. Method Details specifies 16 slots, navigation supervision and a frozen executor with learned projection. README therefore uses the paper's Introduction figure only.
- Method prose describes explicit facts and subtask progress broadly; the appendix gives a deterministic recent action/observation record. The default implements that concrete template; the evidence-checked structured update API is optional and requires a benchmark adapter.
- The paper does not fix projector architecture or optimizer hyperparameters. A linear projector, learning rate 1e-4, three epochs, accumulation of eight and next-action validation loss are documented release defaults, not attributed experimental settings.
- Reciprocal-rank constant 60, navigation temperature 0.07 and hidden width 512 come from the supplied implementation. They are implementation settings beyond the paper's reported table.
- The 1,200-token budget currently bounds skill bodies, as in the supplied code; the explicit record and reference headers are separate. Appendix sensitivity prose mentions the record and skill together, so whether the full injected text must share C_K remains ambiguous.
- The appendix's shared ALFWorld prompt permits `SkillRequest` outputs, whereas the algorithm and released proxy acquire skills automatically at every decision point and do not handle request-triggered commands. The archived template is retained for traceability; use `prompts/alfworld_jitskill_executor.txt` with this proxy.
- The paper does not report the compression-loss coefficients, probe construction, contrast queue size, or a precise centering rule for the contrast anchor. The release defaults to unit reconstruction/alignment/contrast weights, zero probe weight, a queue of 64, and frozen mean-history anchors. These are executable reference choices, not claimed paper hyperparameters.
- Unit tests can verify component contracts, but the released archive contains no pretrained projector/checkpoints, benchmark harnesses, or result artifacts. The corrected implementation needs retraining and end-to-end benchmark validation before claiming score reproduction.

## Anonymous distribution

Release files contain no author/contact/citation block, private server addresses, account names, absolute deployment paths, model credentials, or private dataset dumps. Create a fresh Git history with a neutral contributor identity. Keep the origin repository private; an anonymous review link should expose a snapshot without Git history or origin metadata. Its behavior depends on the separate anonymous hosting service and must be checked after that link is created.

## Validation

Nine CPU tests pass for projection gradients and checkpoint round-trip, frozen decoder parameters, causal before-decision data construction, shared train/serve record format, recurrent gradient truncation, skill budget/mask behavior, loss functions, appendix template text/injection location, native tool-call evidence, and retry-safe recurrence. Python compilation and shell syntax checks pass. README Figure 1 is an unchanged copy of the paper image; visual inspection and metadata inspection found no personal identifiers. Full GPU training, benchmark evaluation and live vLLM embedding-token parity have not been run.
