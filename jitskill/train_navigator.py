# -*- coding: utf-8 -*-
"""Train navigator."""
import argparse
import json
import os
import re

import numpy as np
import torch

from .explicit_record import ExplicitRecord, delta_text
from .latent_state import LatentEncoder, episode_forward
from .library import load_pool
from .navigator import CrossScorer, Navigator, skill_logits
from .objectives import l_nav

API_RE = re.compile(r"\bapis\.([a-z_]+)\.([a-z_0-9]+)\s*\(")
# APIs every AppWorld episode calls; they identify no skill
UBIQUITOUS = {"api_docs.show_api_doc", "api_docs.show_api_descriptions",
              "supervisor.show_account_passwords", "supervisor.complete_task"}


def step_chunks(rec):
    """Decision t sees transition t-1, never its own target action or later feedback.

    Input contract: steps[t].env_in is the observation available BEFORE code[t];
    feedback for code[t-1] is therefore steps[t].env_in.
    """
    out = [""]
    for i in range(1, len(rec["steps"])):
        prev = rec["steps"][i - 1]
        response = (prev.get("say", "") + "\n```python\n" + prev.get("code", "") + "\n```").strip()
        out.append(delta_text(response, rec["steps"][i].get("env_in", "")))
    return out[:len(rec["steps"])]


def build_episodes(traj, cards):
    by_api = {}
    for i, c in enumerate(cards):
        for a in c.get("apis", []):
            by_api.setdefault(a, set()).add(i)
    eps = []
    for rec in traj:
        record, marks = ExplicitRecord(rec["instruction"]), []
        for i, st in enumerate(rec["steps"]):
            if i:
                prev = rec["steps"][i - 1]
                response = prev.get("say", "") + "\n```python\n" + prev.get("code", "") + "\n```"
                record.observe(response, st.get("env_in", ""))
            apis = {"%s.%s" % pair for pair in API_RE.findall(st.get("code", ""))} - UBIQUITOUS
            gold = set().union(*(by_api.get(a, set()) for a in apis)) if apis else set()
            marks.append({"gold": sorted(gold), "bt": record.render()})
        eps.append({"scenario": rec["scenario"], "task_id": rec["task_id"],
                    "chunks": step_chunks(rec), "marks": marks})
    return eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--skills", required=True)
    ap.add_argument("--docs_dir", default="",
                    help="AppWorld api_docs dir; with --neg_pool all its cards join the pool")
    ap.add_argument("--model", required=True, help="encoder base (Qwen3-8B)")
    ap.add_argument("--emb_model", required=True, help="all-MiniLM-L6-v2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr_enc", type=float, default=5e-5)
    ap.add_argument("--lr_nav", type=float, default=1e-3)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--val_scenarios", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0, help="selects the held-out scenarios")
    ap.add_argument("--init_seed", type=int, default=None,
                    help="weight init and episode order; defaults to --seed")
    ap.add_argument("--neg_pool", choices=["skills", "all"], default="skills",
                    help="softmax over the procedural skills only, or over the whole pool")
    ap.add_argument("--head", choices=["pointer", "cross"], default="cross")
    ap.add_argument("--no_anchor_feat", action="store_true",
                    help="cross head: drop the cos(e(B_t), e(k)) feature")
    ap.add_argument("--init_encoder", default="",
                    help="optional train_compression checkpoint used to initialize the encoder")
    a = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    init_seed = a.seed if a.init_seed is None else a.init_seed
    torch.manual_seed(init_seed)
    proc = [c for c in load_pool(a.skills) if c["layer"] == "procedural"]
    if a.neg_pool == "all":
        cards = load_pool(a.skills, docs_dir=a.docs_dir or None)
        gold_off = len(cards) - len(proc)
        assert [c["id"] for c in cards[gold_off:]] == [c["id"] for c in proc]
    else:
        cards, gold_off = proc, 0

    st = SentenceTransformer(a.emb_model, device="cpu")
    card_emb = torch.tensor(st.encode([c["text"] for c in cards], convert_to_numpy=True,
                                      normalize_embeddings=True),
                            dtype=torch.float32, device=a.device)

    traj = [json.loads(l) for l in open(a.traj, encoding="utf-8")]
    eps = build_episodes(traj, proc)
    bt_all = st.encode([m["bt"] for e in eps for m in e["marks"]], batch_size=64,
                       convert_to_numpy=True, normalize_embeddings=True,
                       show_progress_bar=False)
    off, bt_emb = 0, {}
    for e in eps:
        bt_emb[e["task_id"]] = torch.tensor(bt_all[off:off + len(e["marks"])],
                                            dtype=torch.float32, device=a.device)
        off += len(e["marks"])

    enc = LatentEncoder(a.model, a.device)
    if a.init_encoder:
        enc.load_trainable_checkpoint(a.init_encoder)
    nav = Navigator(enc.d_z, card_emb.shape[1], card_emb.shape[1]).to(a.device)
    head = None
    if a.head == "cross":
        head = CrossScorer(card_emb.shape[1], use_anchor=not a.no_anchor_feat).to(a.device)

    tok_cache = {e["task_id"]: [enc.tok(c, add_special_tokens=False)["input_ids"]
                                for c in e["chunks"]] for e in eps}

    scen = sorted(set(e["scenario"] for e in eps))
    np.random.RandomState(a.seed).shuffle(scen)
    if not 0 < a.val_scenarios < len(scen):
        raise ValueError("validation must leave at least one training scenario")
    val = set(scen[:a.val_scenarios])
    rng = np.random.RandomState(init_seed)
    tr_eps = [e for e in eps if e["scenario"] not in val]
    va_eps = [e for e in eps if e["scenario"] in val]
    gold_sizes = [len(m["gold"]) for e in eps for m in e["marks"] if m["gold"]]
    mean_gold = sum(gold_sizes) / max(1, len(gold_sizes))
    print("[nav] %d episodes (%d train / %d val), %d skills, pool=%d, chance R@1=%.4f"
          % (len(eps), len(tr_eps), len(va_eps), len(proc), len(cards),
             mean_gold / len(cards)), flush=True)

    def score(z, b, E=None):
        return skill_logits(nav, head, a.head, z, b, card_emb if E is None else E)

    enc_params = [p for p in enc.parameters() if p.requires_grad]
    groups = [{"params": enc_params, "lr": a.lr_enc},
              {"params": list(nav.parameters()), "lr": a.lr_nav}]
    params = enc_params + list(nav.parameters())
    if head is not None:
        groups.append({"params": list(head.parameters()), "lr": a.lr_head})
        params += list(head.parameters())
    opt = torch.optim.AdamW(groups, weight_decay=1e-2)

    swap_perm = torch.tensor(np.random.RandomState(12345).permutation(len(cards)),
                             device=a.device)

    def tok_steps(e):
        return tok_cache[e["task_id"]]

    def evaluate(which, swapped=False):
        enc.eval(); nav.eval()
        if head is not None:
            head.eval()
        hit = {1: 0, 3: 0, 5: 0}; n = 0
        E = card_emb[swap_perm] if swapped else card_emb
        with torch.no_grad():
            for e in which:
                for i, z in episode_forward(enc, tok_steps(e)):
                    g = e["marks"][i]["gold"]
                    if not g:
                        continue
                    gs = set(gold_off + x for x in g)
                    s = score(z.mean(0, keepdim=True), bt_emb[e["task_id"]][i:i + 1], E)[0]
                    r = s.argsort(descending=True).tolist()
                    n += 1
                    for k in hit:
                        if gs & set(r[:k]):
                            hit[k] += 1
        return n, {k: v / max(1, n) for k, v in hit.items()}

    if not any(m["gold"] for e in va_eps for m in e["marks"]):
        raise ValueError("validation has no supervised decisions")
    best = -1.0
    for ep_i in range(a.epochs):
        enc.train(); nav.train()
        if head is not None:
            head.train()
        tot, cnt = 0.0, 0
        for j, idx in enumerate(rng.permutation(len(tr_eps)), 1):
            e = tr_eps[idx]
            opt.zero_grad()
            ep_loss, n_sup = 0.0, 0
            for i, z in episode_forward(enc, tok_steps(e)):
                g = e["marks"][i]["gold"]
                if not g:
                    continue
                loss = l_nav(score(z.mean(0, keepdim=True), bt_emb[e["task_id"]][i:i + 1]),
                             g, gold_off)
                loss.backward()                  # per step: each step holds its own 8B forward
                ep_loss += float(loss); n_sup += 1
            if not n_sup:
                continue
            gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
            if not torch.isfinite(gn):
                opt.zero_grad(set_to_none=True)
                continue
            opt.step()
            tot += ep_loss / n_sup; cnt += 1
            if j % 10 == 0:
                print("[nav] ep%d %d/%d loss %.4f" % (ep_i + 1, j, len(tr_eps),
                                                     tot / max(1, cnt)), flush=True)
        n, h = evaluate(va_eps)
        _, hs = evaluate(va_eps, swapped=True)
        print("[nav] epoch %d train_loss %.4f | VAL n=%d R@1=%.3f R@3=%.3f R@5=%.3f | "
              "SWAPPED R@1=%.3f" % (ep_i + 1, tot / max(1, cnt), n, h[1], h[3], h[5], hs[1]),
              flush=True)
        if h[1] >= best:
            best = h[1]
            os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
            enc.enc.save_pretrained(a.out + "_enc")
            torch.save({"head": a.head, "no_anchor_feat": a.no_anchor_feat,
                        "head_state": head.state_dict() if head is not None else None,
                        "nav": nav.state_dict(), "slots": enc.slots.detach().cpu(),
                        "init_state": enc.init_state.detach().cpu(),
                        "proj": enc.proj.state_dict(),
                        "d_z": enc.d_z, "d_txt": card_emb.shape[1], "val_r1": best,
                        "neg_pool": a.neg_pool, "seed": a.seed, "init_seed": init_seed,
                        "n_pool": len(cards), "n_proc": len(proc)}, a.out + "_nav.pt")
            print("[nav] saved (val R@1 %.3f)" % best, flush=True)
    print("[nav] DONE best val R@1 %.3f" % best)


if __name__ == "__main__":
    main()
