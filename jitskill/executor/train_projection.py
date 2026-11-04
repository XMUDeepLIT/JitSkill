"""Train only P_psi on recorded next actions, with a frozen executor (Eq. executor-training)."""
import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
from .projection import StateProjector
from .latent_prefix import SENTINEL
from ..objectives import l_exec


def render(tok, messages):
    msgs = [dict(m) for m in messages]
    for i in range(len(msgs) - 1, -1, -1):
        if msgs[i].get("role") == "user":
            if not isinstance(msgs[i].get("content"), str):
                raise ValueError("training messages must contain text, not serialized latents")
            msgs[i]["content"] += SENTINEL
            break
    else:
        raise ValueError("no user turn for state-token insertion")
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)


def splice_inputs(embedding, prompt_ids, target_ids, marker_id, projected):
    if prompt_ids.count(marker_id) != 1:
        raise ValueError("exactly one latent marker must appear in prompt")
    pos = prompt_ids.index(marker_id)
    ids = prompt_ids[:pos] + prompt_ids[pos + 1:] + target_ids
    x = embedding(torch.tensor([ids], device=projected.device))
    return torch.cat([x[:, :pos], projected.to(x.dtype).unsqueeze(0), x[:, pos:]], dim=1)


def main():
    ap = argparse.ArgumentParser()
    for name in ("sft", "latents", "model", "out"):
        ap.add_argument("--" + name, required=True)
    ap.add_argument("--epochs", type=int, default=3)
    # Projector optimizer choices are release defaults; the paper does not report them.
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max_len", type=int, default=16384)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--holdout", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    if a.accum < 1 or a.epochs < 1:
        raise ValueError("accum and epochs must be positive")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.add_special_tokens({"additional_special_tokens": [SENTINEL]})
    marker = tok(SENTINEL, add_special_tokens=False)["input_ids"]
    if len(marker) != 1:
        raise ValueError("latent marker must tokenize atomically")
    rows = [json.loads(line) for line in Path(a.sft).read_text().splitlines() if line.strip()]
    z = np.load(a.latents, mmap_mode="r")
    if z.ndim != 3 or z.shape[1] != 16 or len(z) != len(rows) or not len(rows):
        raise ValueError("expected nonempty row-aligned latents of shape (n,16,d_z)")
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).to(a.device)
    model.requires_grad_(False)
    model.eval()  # no executor dropout or LoRA adaptation
    model.config.use_cache = False
    embedding = model.get_input_embeddings()
    projector = StateProjector(z.shape[2], embedding.weight.shape[1]).to(a.device)
    optimizer = torch.optim.AdamW(projector.parameters(), lr=a.lr)
    scenarios = sorted({r.get("scenario") or r["task_id"].split("_")[0] for r in rows})
    if not 0 < a.holdout < len(scenarios):
        raise ValueError("holdout must leave both training and validation scenarios")
    random.Random(a.seed).shuffle(scenarios)
    held = set(scenarios[:a.holdout])
    train, val = [], []
    for i, row in enumerate(rows):
        scenario = row.get("scenario") or row["task_id"].split("_")[0]
        (val if scenario in held else train).append(i)

    def loss_one(i):
        row = rows[i]
        prompt_ids = tok(render(tok, row["messages"]), add_special_tokens=False)["input_ids"]
        target = tok(row["target"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        if len(prompt_ids) - 1 + z.shape[1] + len(target) > a.max_len:
            return None
        state = torch.tensor(np.array(z[i]), dtype=torch.float32, device=a.device)
        x = splice_inputs(embedding, prompt_ids, target, marker[0], projector(state))
        # Frozen weights remain in autograd so action loss can reach the projector.
        output = model(inputs_embeds=x, use_cache=False)
        return l_exec(output.logits[0, -len(target)-1:],
                      torch.tensor(target, device=a.device))

    best = float("inf")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(a.epochs):
        order = train.copy()
        random.Random(a.seed + epoch).shuffle(order)
        optimizer.zero_grad(set_to_none=True)
        pending, count, total = 0, 0, 0.0
        def update(n):
            for parameter in projector.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(n)
            norm = torch.nn.utils.clip_grad_norm_(projector.parameters(), 1.0)
            if not torch.isfinite(norm):
                raise FloatingPointError("non-finite projection gradients")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        for i in order:
            loss = loss_one(i)
            if loss is None:
                continue
            loss.backward()
            pending += 1; count += 1; total += float(loss.detach())
            if pending == a.accum:
                update(pending); pending = 0
        if pending:
            update(pending)
        if not count:
            raise ValueError("all training rows exceed max_len")
        losses = []
        with torch.no_grad():
            for i in val:
                loss = loss_one(i)
                if loss is not None:
                    losses.append(float(loss))
        if not losses:
            raise ValueError("all validation rows exceed max_len")
        mean = sum(losses) / len(losses)
        if mean < best:
            best = mean
            projector.save(a.out)
        assert all(p.grad is None and not p.requires_grad for p in model.parameters())
        print(f"epoch={epoch+1} train_loss={total/count:.4f} val_loss={mean:.4f}", flush=True)

if __name__ == "__main__":
    main()
