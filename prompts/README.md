`alfworld_executor.txt` is the shared ALFWorld template reproduced from the paper appendix. It retains `SkillRequest` syntax for the baseline protocol and is included as an archival reference.

`alfworld_jitskill_executor.txt` is the runtime template for this release. The proxy retrieves and injects skills automatically at every decision point, so the executor must emit an environment `Action` rather than a `SkillRequest`; the released proxy does not implement request-triggered retrieval commands.

The progress record is extracted by deterministic rules, without a language-model prompt or call. AppWorld, SkillsBench and tau2-Bench use their benchmark executor prompts; this release does not invent replacement system prompts.
