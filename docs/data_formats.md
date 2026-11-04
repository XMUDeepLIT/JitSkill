# Data contracts

## Skill library

JSONL, one skill per row. Minimal retrieval card:

```json
{"id":"skill::inspect","layer":"procedural","text":"Inspect the required fields before updating an object.","apis":["objects.inspect"]}
```

A structured record with `skill_id`, `name`, `app`, `trigger`, `preconditions`, `procedure`, `failure_handling`, and `apis` is also accepted. Each `procedure` entry contains `step`, `what`, and optional `code`. `apis` provides the mapping from an executed operation to the next-step skill supervision target. Decision points with no matching skill are not supervised, but their evidence remains in the recurrent history.

SkillsBench cards may add `inject`: a compact name, purpose, and mounted `SKILL.md` path. Use portable mount paths. Declarative API cards can optionally join the scoring pool; they are never selected as procedural skills.

## Navigation trajectories

```json
{"task_id":"scenario_variant","scenario":"scenario","instruction":"Update an object","steps":[{"say":"Inspect first","code":"apis.objects.inspect()","env_in":"Initial observation"},{"say":"Update it","code":"apis.objects.update()","env_in":"Feedback from inspect"}]}
```

**`env_in` is the observation available before the code in that row is executed.** At decision t, the model reads only completed transitions preceding t; the current row's code supplies labels but never input history. Convert logs with post-action observations into this format explicitly. Terminal feedback has no next-action target.

## Projector corpus

The builder reads raw recorded calls at `tasks/<task_id>/logs/lm_calls.jsonl`, each with `input.messages` and `output.choices[0].message.content`, plus a JSON list of successful official-training task IDs. Already-injected requests are rejected. Output rows contain `task_id`, `messages`, `target`, `retrieved`, and `n_tokens`. A corresponding `.npy` contains raw encoder-space states of shape `(number_of_rows, 16, d_z)` in the same row order.

For history-compression pretraining, trajectory JSONL may additionally provide a scalar `trajectory_weight` and a per-decision `compression_probe` list. Both default to one and zero, respectively, when absent. The paper specifies the weighted objective but not numerical weights; the released training command's defaults are implementation choices, not reported experimental settings.

The current corpus adapter is for AppWorld-shaped user-turn logs. Other benchmark adapters must preserve the same before-action timing, injection location and scenario-disjoint split. None of these files contain the recorded benchmark trajectories in this release.

## Runtime history

Pass the full raw message history and a unique per-episode `lk` key. AppWorld worked demonstrations are excluded using the actual-task marker. Assistant actions plus subsequent user/tool observations form transitions. The initial user observation supplies a generic task goal; adapters with different task layouts must supply the correct task statement. Do not feed previously injected blocks back into the proxy.
