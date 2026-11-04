# Paper-to-code mapping

JitSkill's Method and Method Details map to the following implementation contracts.

| Paper component | Code | Contract |
|---|---|---|
| Explicit task progress, Eq. explicit-progress | `jitskill/explicit_record.py` | Goal and six recent actions/failures use appendix limits; action outcomes come from observed feedback |
| Structured facts/subtask progress | `ExplicitRecord.observe` | Adapter-supplied values require direct evidence in the observation |
| Latent update, Eq. latent-state | `jitskill/latent_state.py` | Frozen encoder base plus LoRA, initial state and 16 slots; preceding transitions only |
| Truncated recurrent training | `episode_forward` | Forward state persists; gradients detach between updates |
| History-compression pretraining, Eq. memory-training | `jitskill/train_compression.py` | Trains encoder LoRA, slots and recurrent projection; saves the final checkpoint without validation selection |
| Navigation, Eq. nav-training | `jitskill/train_navigator.py`, `navigator.py` | Requires `--init_encoder` from compression; uniform labels over next-action supporting skills jointly optimize encoder and navigator |
| Query and description scoring | `Navigator`, `CrossScorer` | Joint input of mean latent slots and frozen record embedding; description-conditioned cross head for new libraries |
| Skill selection | `jitskill/fusion.py` | Learned/current-step dense scores and BM25 fused by reciprocal ranks, procedural mask, top-k token packing |
| Mapping, Eq. latent-state-projection | `jitskill/executor/projection.py` | Linear P_psi maps d_z to the executor embedding dimension |
| Frozen executor likelihood, Eq. executor-training | `train_projection.py` | Only P_psi is optimized; executor parameters remain frozen |
| History reconstruction / JSD / contrast / weighting | `jitskill/objectives.py`, `train_compression.py` | Weighted encoder-pretraining objective |
| Skill injection and shared state | `serving/inject_proxy.py` | Same B_t and Z_t used for navigation and execution; only Z_t is projected for executor |
| Prompt templates | `prompts/` | Archived appendix template and automatic-proxy runtime template use the paper's injected-block text |

## Training and checkpoint selection

- **Compression:** `scripts/train_compression.sh` saves the encoder after the final training epoch. There is no validation split or validation-based checkpoint selection in this stage.
- **Navigator:** both `scripts/train_navigator.sh` and the Python CLI require the compression checkpoint. Five scenario-disjoint training scenarios are held out by default; checkpoint selection maximizes validation recall@1.
- **Projector:** five scenario-disjoint training scenarios are held out by default; checkpoint selection minimizes validation next-action loss. Only the linear projector is trained.
- **Teacher calls:** `work/teacher` is supplied by an external benchmark harness and logging adapter. The repository does not generate the original executor calls. The corpus builder reads AppWorld-shaped raw user-turn logs; adapters for other benchmarks must preserve before-action timing and injection placement. See [data contracts](data_formats.md).

## Runtime controls and implementation settings

- The launch script requires `ALPHA`; the proxy CLI requires `--alpha`. ALFWorld uses 0.75. Supply the training-selected weight explicitly for other benchmarks.
- The request `model` string carries proxy controls, e.g. `lk=episode-1::k=3::bud=1200`. `lk` must be nonempty and unique per episode, including reruns. Missing or blank keys receive HTTP 400 before state updates. The upstream executor model ID comes from `--upstream_model` (`MODEL_NAME` in the launch script), not this control string.
- The proxy folds unseen history transitions once per episode. Repeated raw histories do not duplicate recurrence.
- Injection locations are `user` for ALFWorld/AppWorld, `tool` for SkillsBench, and `system` for tau2-Bench. SkillsBench adapters must supply an injectable tool-result message on every decision, including the initial request. Missing tool results receive HTTP 400 before state updates.
- Projector defaults: linear map, learning rate 1e-4, three epochs, gradient accumulation of eight, validation by next-action loss.
- Retrieval defaults: reciprocal-rank constant 60, navigation temperature 0.07 and hidden width 512.
- Compression defaults: unit reconstruction/alignment/contrast weights, zero probe weight, a contrast queue of 64 and frozen mean-history anchors.
- The current token budget bounds selected skill bodies; explicit records and reference headers are added separately.
- The archived ALFWorld appendix prompt permits `SkillRequest`. The automatic proxy retrieves before each action decision; use `prompts/alfworld_jitskill_executor.txt` for that runtime.

## README figure provenance

The README displays **two** illustrations. Sources below refer to Overleaf revision `dcf2365` of project `6a9682feb0dc96690bb7af8f`.

| README asset | Source | Relationship to LaTeX |
|---|---|---|
| `assets/intro_retrieval.png` | Page 1 exported from `resources/intro.pdf` | The current Introduction instead includes `resources/skilljit_intro1.png`. The README export is not an unchanged copy of that PNG. |
| `assets/jitskill_pipeline.png` | `resources/skilljit_method.png` | Byte-identical to the image included by the current Method section. Git blob: `2cd76cc50128f1445697aa0e208275431d4eb38c`. |

The README Introduction export has SHA-256 `acbe91bf06487b523e96ed0e8acc6395bd3d9ec4e91e171cfe11df0efc073cd9`. Its PDF source has Git blob `2ad095d366e7b10dee59b7a9002dd32f4bbb1606`; the LaTeX Introduction PNG has Git blob `4b3653c38ac9a8a87b1f2a612ee8cc5fa97b2d09`. These identify distinct source assets. Keep both entries when updating figures, and record any changed source, page or rendering operation. `assets/intro.png` is a legacy asset and is not displayed by the current README.

## Release contents and checks

Included: compression, navigation and projector training; progress tracking; skill selection; executor proxy; prompt templates; component tests. External inputs: benchmark harnesses and adapters, model weights, pretrained checkpoints, raw trajectories and result logs.

The CPU test suite checks projection gradients and serialization, frozen decoder parameters, before-decision data construction, shared train/serve records, recurrent gradient truncation, skill budgets and masks, losses, prompt injection locations, tool-call evidence, retry-safe recurrence, episode-key validation and upstream model forwarding. These are component checks; benchmark runs use the external harnesses and inputs above.
