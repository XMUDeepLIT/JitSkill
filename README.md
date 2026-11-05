<div align="center">

# JitSkill
### Just-in-Time Skill Retrieval via Long-Horizon State Tracking

**[Paper](https://arxiv.org/abs/2611.01701)** · **[Method](#method)** · **[Installation](#installation)** · **[Training](#training)** · **[Serving](#serving)** · **[Paper-to-code mapping](docs/paper_alignment.md)**

</div>

Large Language Model (LLM)-based agents can reuse procedural knowledge through external skill libraries. Existing retrieval methods, however, typically choose a fixed skill set before execution. This is restrictive for long-horizon interaction, where intermediate observations, execution outcomes, and newly revealed constraints expose procedural needs that the initial instruction does not specify.

## Just-in-Time Skill Retrieval

<p align="center"><img src="assets/intro_retrieval.png" width="100%" alt="Paper introduction figure: pre-execution and just-in-time skill retrieval"></p>

*Introduction illustration exported from page 1 of `resources/intro.pdf` in the paper's Overleaf project. Pre-execution retrieval can miss needs revealed during execution, whereas just-in-time retrieval acquires skills on demand as needs become observable.*

JitSkill couples this execution-time retrieval paradigm with task-state tracking. A candidate skill should be judged not only by the current observation, but also by prior execution: what has been completed, which results can be reused, which attempts have failed, and which requirements remain unresolved. This avoids **execution drift**, in which locally plausible skill decisions repeat completed work or conflict with accumulated task progress.

## Method

<p align="center"><img src="assets/jitskill_pipeline.png" width="100%" alt="Paper overview of JitSkill task-progress tracking and progress-grounded skill decisions"></p>

*Method overview from the paper. Explicit records and latent representations jointly track task progress, guiding both skill retrieval and action generation.*

JitSkill maintains an evolving task-progress state **S_t = (B_t, Z_t)**. The explicit record **B_t** organizes inspectable execution evidence, including operations, outcomes, and failure feedback. The fixed-size latent state **Z_t** recurrently carries forward historical context that is difficult to enumerate as a small set of textual fields. The same state grounds both skill retrieval and subsequent execution.

1. **Track progress.** A recurrent encoder updates latent slots from new action--feedback evidence, while the explicit record retains directly inspectable task facts.
2. **Acquire skills.** At each decision point, a navigator combines the tracked state with current execution context to select the skill subset needed next from the library.
3. **Use skills.** A frozen executor receives the current context, selected skills, explicit record, and projected latent state tokens to generate the next action.
4. **Update.** The action and environment feedback update task progress, making their consequences available to later skill decisions.

Training proceeds in three stages. The encoder is first pretrained with a history-compression objective, then jointly fine-tuned with the navigator using next-step skill supervision, and finally used to train the executor-side projection. A description-conditioned scoring head supports changes to the skill library without retraining.

## Results

Across SkillsBench, ALFWorld, and $\tau^2$-Bench, JitSkill achieves the highest overall task-success rates reported in the paper. On SkillsBench, it improves over the strongest execution-time retrieval baseline by **20.6%** relatively. See the [paper](https://arxiv.org/abs/2611.01701) for experimental protocols, baselines, and analyses.

## Installation

Requirements:

- Python 3.10 or newer; Linux and Bash for the launch scripts.
- NVIDIA GPUs and a CUDA-compatible PyTorch installation for model training and serving.
- Local model weights: Qwen3-8B, all-MiniLM-L6-v2, and the frozen executor (Qwen3.8-27B in the paper).
- The official benchmark environment, skill library, and training trajectories described below.

Clone the repository and install in a virtual environment:

```bash
git clone https://github.com/XMUDeepLIT/JitSkill.git
cd JitSkill
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m unittest discover -s tests -v
```

For serving, install the vLLM extra in the serving environment:

```bash
python -m pip install -e ".[serving]"
```

The vLLM build must support chat `prompt_embeds` and `--enable-prompt-embeds`. GPU memory requirements depend on the executor, precision, context length, and whether the navigator shares its GPU. The serving example below places the executor and navigator on separate GPUs. Keep model weights, credentials, benchmark data, and generated checkpoints outside version control.

## Training

The training lifecycle is:

```text
History compression -> Navigation fine-tuning -> Executor projection
```

### Prepare the inputs

Prepare AppWorld official-training trajectories and the AppWorld skill library in the [data formats](docs/data_formats.md). Each trajectory records the task instruction, scenario, and before-action observations; skill cards map executed APIs to supporting skills.

```bash
export ENCODER_BASE=/path/to/Qwen3-8B
export EMBEDDER=/path/to/all-MiniLM-L6-v2
export TRAIN_TRAJECTORIES=/path/to/appworld_train.jsonl
export SKILLS=/path/to/appworld_skills.jsonl
export EXECUTOR_DIR=/path/to/frozen-executor
export WORK_DIR=work
mkdir -p "$WORK_DIR"
```

### 1. Pretrain the history compressor

```bash
bash scripts/train_compression.sh
export COMPRESSOR_CKPT="$WORK_DIR/compression.pt"
```

This stage saves the final encoder checkpoint. It does not use validation-based checkpoint selection.

### 2. Fine-tune the navigator

```bash
export HEAD=pointer  # AppWorld; use cross for a different target skill library
bash scripts/train_navigator.sh
```

The navigator requires the compression checkpoint and jointly trains the encoder and skill-scoring components. `HEAD=cross` trains the description-conditioned head on the same AppWorld trajectories for use with another skill library. The checkpoint is selected by recall@1 on five scenario-disjoint held-out training scenarios.

### 3. Train the executor projection

First, use an **external benchmark harness and logging adapter** to run the executor on official training tasks. Supply raw calls recorded before JitSkill injection and a list of successful task IDs:

```text
/path/to/teacher_run/
├── successes.json
└── tasks/<task_id>/logs/lm_calls.jsonl
```

Each log entry contains `input.messages` and `output.choices[0].message.content`. This directory is an external input, not generated by the training scripts. The supplied corpus builder reads AppWorld-shaped user-turn logs; other benchmarks require an adapter preserving their action format and injection location.

Set `ALPHA` explicitly to the fusion weight selected for the teacher benchmark on training data. For this stage, `SKILLS` must point to the corresponding teacher skill library; use the same library, scoring head, and fusion weight when serving that executor.

```bash
export TEACHER_DIR=/path/to/teacher_run
: "${ALPHA:?set the fusion weight selected for the teacher benchmark}"

python -m jitskill.executor.build_sft_corpus \
  --exp "$TEACHER_DIR" --successes "$TEACHER_DIR/successes.json" \
  --out "$WORK_DIR/actions.jsonl" --latent_out "$WORK_DIR/states.npy" \
  --tokenizer "$EXECUTOR_DIR" --skills "$SKILLS" \
  --emb_model "$EMBEDDER" --nav_base "$ENCODER_BASE" \
  --nav_ckpt "$WORK_DIR/nav_nav.pt" --enc_dir "$WORK_DIR/nav_enc" \
  --nav_device cuda:0 --alpha "$ALPHA"

export SFT_ROWS="$WORK_DIR/actions.jsonl"
export LATENT_ROWS="$WORK_DIR/states.npy"
bash scripts/train_projection.sh
```

Only the projection is optimized; executor parameters remain frozen. Checkpoint selection minimizes next-action loss on five scenario-disjoint held-out training scenarios. Task IDs use `<scenario>_<variant>`; the corpus adapter must preserve this grouping.

Training outputs:

| Output | Used by |
|---|---|
| `work/compression.pt` | Navigator initialization |
| `work/nav_enc/` and `work/nav_nav.pt` | Corpus construction and serving |
| `work/actions.jsonl` and `work/states.npy` | Row-aligned projector training inputs |
| `work/projector.pt` | Executor-side state projection |

These paths assume `WORK_DIR=work`. See [implementation settings](docs/paper_alignment.md) for optimizer defaults and the equation-to-code mapping.

## Serving

Serving has two processes: a vLLM executor and the JitSkill proxy. Start each from the repository root in its own activated environment. The commands below place the executor on GPU 0 and the navigator on GPU 1.

### 1. Start the frozen executor

Set `REASONING_PARSER` and `TOOL_CALL_PARSER` to the executor's official parser names supported by your vLLM installation. Configure the precision and context length explicitly:

```bash
export EXECUTOR_DIR=/path/to/frozen-executor

CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model "$EXECUTOR_DIR" --served-model-name executor \
  --quantization fp8 --max-model-len 32768 --enable-prompt-embeds \
  --reasoning-parser "${REASONING_PARSER:?set the executor reasoning parser}" \
  --enable-auto-tool-choice \
  --tool-call-parser "${TOOL_CALL_PARSER:?set the executor tool-call parser}" \
  --host 127.0.0.1 --port 8000
```

Wait until `http://127.0.0.1:8000/health` returns HTTP 200 before starting the proxy.

### 2. Start the JitSkill proxy

In the second terminal, set the model paths and the benchmark's skill library. The following example is for **ALFWorld**; its fusion weight is 0.75. For another benchmark, explicitly set its training-selected weight and injection role from the table below.

```bash
export EXECUTOR_DIR=/path/to/frozen-executor
export NAV_BASE=/path/to/Qwen3-8B
export EMBEDDER=/path/to/all-MiniLM-L6-v2
export WORK_DIR=work
export SKILLS=/path/to/alfworld_skills.jsonl
export ALPHA=0.75
export INJECTION_ROLE=user

CUDA_VISIBLE_DEVICES=1 python -m serving.inject_proxy \
  --upstream http://127.0.0.1:8000 --upstream_model executor \
  --tokenizer "$EXECUTOR_DIR" --skills "$SKILLS" --emb_model "$EMBEDDER" \
  --nav_base "$NAV_BASE" --nav_ckpt "$WORK_DIR/nav_nav.pt" \
  --enc_dir "$WORK_DIR/nav_enc" --projector "$WORK_DIR/projector.pt" \
  --nav_device cuda:0 --alpha "$ALPHA" --injection_role "$INJECTION_ROLE" \
  --host 127.0.0.1 --port 8100
```

`cuda:0` here refers to GPU 1, the only device visible to the proxy process. Use a navigator with the appropriate scoring head and a projector trained for the selected executor.

| Benchmark | `INJECTION_ROLE` | Adapter input |
|---|---|---|
| ALFWorld / AppWorld | `user` | Latest environment observation as a user message |
| SkillsBench | `tool` | An existing injectable tool result on every request, including the initial decision |
| tau2-Bench | `system` | Domain system prompt; the block is appended to it |

SkillsBench cards can carry a compact `inject` pointer with the skill name, purpose, and mounted `SKILL.md` path. Setting `INJECTION_ROLE=tool` alone does not create a tool result; the benchmark adapter supplies that message. The proxy returns HTTP 400 if it is absent. `ALPHA` has no launch-script default, and `--alpha` is required by the proxy CLI; select other benchmark weights on designated training data, not evaluation tasks.

### 3. Connect the benchmark harness

Configure the external harness with:

| Setting | Value |
|---|---|
| OpenAI-compatible base URL | `http://127.0.0.1:8100/v1` |
| Request model | `lk=<unique-episode-id>::k=3::bud=1200` |
| History | Full raw messages before JitSkill injection |
| Streaming | `false` |

The request model string contains **proxy controls**, not a model-weight identifier: `lk` identifies the episode, `k` sets the skill count, and `bud` sets the skill-body token budget. Use a new `lk` for every episode and rerun; missing or blank keys receive HTTP 400. Repeating the same raw history does not advance latent state again. The proxy forwards requests to the engine using the separately configured `--upstream_model executor`.

For ALFWorld's automatic-retrieval proxy, use [`prompts/alfworld_jitskill_executor.txt`](prompts/alfworld_jitskill_executor.txt). The archived appendix prompt is in [`prompts/alfworld_executor.txt`](prompts/alfworld_executor.txt).

The combined launcher [`serving/serve_executor.sh`](serving/serve_executor.sh) starts both processes on one host. Use the separate commands above to configure FP8, context length, model parsers, and GPU placement explicitly.

## Release scope

| Included in this repository | Obtain or prepare separately |
|---|---|
| Explicit records and recurrent latent-state components | Encoder, embedder, and executor model weights |
| Compression, navigation, and projector training commands | Official-training trajectories and teacher executor-call logs |
| Skill-library loader and retrieval components | Benchmark skill libraries in the documented JSONL format |
| OpenAI-compatible executor proxy | Official benchmark harnesses, environments, and logging/runtime adapters |
| Prompt templates, component tests, and paper-to-code mapping | Pretrained JitSkill checkpoints and benchmark result logs |

Benchmark harnesses, pretrained checkpoints, raw trajectories, and result logs are not bundled. Run the component tests with `python -m unittest discover -s tests -v`; run benchmark evaluation through the external harness configured in [Serving](#serving).

See [data contracts](docs/data_formats.md) for input schemas, [`prompts/`](prompts/) for runtime and archived templates, and [paper alignment](docs/paper_alignment.md) for implementation settings and the sources of both README illustrations.
