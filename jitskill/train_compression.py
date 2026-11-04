# -*- coding: utf-8 -*-
"""History-compression pretraining for the recurrent latent encoder.

This command writes a trainable-only checkpoint consumable by ``train_navigator
--init_encoder``.  It never trains the frozen encoder trunk or an executor.
"""
import argparse
import collections
import json
import os
import random

import torch

from .latent_state import DELTA_MAX, ENC_MAX, LatentEncoder
from .objectives import l_alignment, l_compression, l_contrast, l_reconstruction
from .train_navigator import step_chunks


def _episodes(path, tokenizer):
    """Return scenario-tagged transition-token sequences from training trajectories."""
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    out = []
    for row in rows:
        chunks = step_chunks(row)
        ids = [tokenizer(chunk, add_special_tokens=False)["input_ids"][-DELTA_MAX:]
               for chunk in chunks]
        probes = row.get("compression_probe", [0.0] * len(ids))
        if len(probes) != len(ids):
            raise ValueError("compression_probe must have one value per decision point")
        out.append({"scenario": row["scenario"], "steps": ids,
                    "weight": float(row.get("trajectory_weight", 1.0)),
                    "probes": [float(v) for v in probes]})
    return out


def _reconstruction_logits(enc, z, target):
    """Teacher-force target tokens after the latent slots and return their causal logits."""
    target_t = torch.tensor(target, device=enc.dev)
    target_emb = enc.emb[target_t].float()
    inputs = torch.cat([z.float(), target_emb], dim=0).unsqueeze(0).to(enc.emb.dtype)
    logits = enc.enc(inputs_embeds=inputs, use_cache=False).logits[0]
    start = z.shape[0] - 1
    return logits[start:start + len(target)]


def _frozen_history_anchor(enc, history):
    """Mean hidden state of the uncompressed history, used as a no-gradient contrast anchor."""
    ids = torch.tensor(history[-ENC_MAX:], device=enc.dev).unsqueeze(0)
    with torch.no_grad():
        output = enc.enc.get_base_model()(input_ids=ids, output_hidden_states=True,
                                          use_cache=False)
        return output.hidden_states[-1][0].float().mean(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True, help="official-training trajectories JSONL")
    ap.add_argument("--model", required=True, help="encoder base model")
    ap.add_argument("--out", required=True, help="output .pt trainable encoder checkpoint")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--lambda_rec", type=float, default=1.0)
    ap.add_argument("--lambda_probe", type=float, default=0.0)
    ap.add_argument("--lambda_align", type=float, default=1.0)
    ap.add_argument("--lambda_disc", type=float, default=1.0)
    ap.add_argument("--queue", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    if min(a.lambda_rec, a.lambda_probe, a.lambda_align, a.lambda_disc) < 0 or a.queue < 1:
        raise ValueError("loss weights must be nonnegative and --queue must be positive")
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    enc = LatentEncoder(a.model, a.device)
    episodes = _episodes(a.traj, enc.tok)
    if not episodes:
        raise ValueError("no training trajectories")
    opt = torch.optim.AdamW([p for p in enc.parameters() if p.requires_grad], lr=a.lr,
                            weight_decay=1e-2)
    anchors = collections.deque(maxlen=a.queue)

    for epoch in range(a.epochs):
        rng.shuffle(episodes)
        total, count = 0.0, 0
        enc.train()
        for episode in episodes:
            opt.zero_grad(set_to_none=True)
            z = enc.init_state.float()
            history = []
            ep_loss, n_terms = 0.0, 0
            for step_i, delta in enumerate(episode["steps"]):
                if not delta:
                    continue
                history.extend(delta)
                z = enc.step(z, delta)
                rec = l_reconstruction(_reconstruction_logits(enc, z, delta),
                                       torch.tensor(delta, device=enc.dev))
                teacher = enc.enc.get_base_model()(input_ids=torch.tensor(
                    history[-ENC_MAX:], device=enc.dev).unsqueeze(0), use_cache=False).logits[0, -1]
                student = enc.enc(inputs_embeds=z.float().unsqueeze(0).to(enc.emb.dtype),
                                  use_cache=False).logits[0, -1]
                align = l_alignment(teacher, student)
                anchor = _frozen_history_anchor(enc, history)
                if anchors:
                    disc = l_contrast(z, anchor, torch.stack(tuple(anchors)).to(enc.dev))
                else:
                    disc = z.sum() * 0.0
                loss = l_compression(rec, align, disc, weight=episode["weight"],
                                     lambda_rec=a.lambda_rec, lambda_probe=a.lambda_probe,
                                     probe=episode["probes"][step_i],
                                     lambda_align=a.lambda_align, lambda_disc=a.lambda_disc)
                loss.backward()
                ep_loss += float(loss.detach())
                n_terms += 1
                anchors.append(anchor.detach().cpu())
                z = z.detach()  # recurrent forward state persists; gradients are truncated.
            if not n_terms:
                continue
            grad = torch.nn.utils.clip_grad_norm_([p for p in enc.parameters() if p.requires_grad], 1.0)
            if not torch.isfinite(grad):
                opt.zero_grad(set_to_none=True)
                continue
            opt.step()
            total += ep_loss / n_terms
            count += 1
        print("[comp] epoch %d loss %.4f" % (epoch + 1, total / max(1, count)), flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    enc.save_trainable_checkpoint(a.out)


if __name__ == "__main__":
    main()
