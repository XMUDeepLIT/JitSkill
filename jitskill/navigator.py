# -*- coding: utf-8 -*-
"""Navigator."""
import threading
import hashlib

import numpy as np
import torch
import torch.nn as nn

from .latent_state import DELTA_MAX, ENC_MAX, N_LATENT, fold  # noqa: F401

TEMPERATURE = 0.07


class Navigator(nn.Module):
    """g_phi: [mean(Z_t) ; e(B_t)] -> unit query vector q_t (the pointer)."""

    def __init__(self, d_z, d_txt, d_out, hidden=512):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d_z + d_txt), nn.Linear(d_z + d_txt, hidden),
                                 nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden, d_out))

    def forward(self, z, t):
        q = self.net(torch.cat([z, t], -1))
        return q / (q.norm(dim=-1, keepdim=True) + 1e-8)


class CrossScorer(nn.Module):
    """s(k) = f(q_t, e(k)): the skill description is an argument of the score.

    Interaction features are relative (elementwise product, absolute difference, and the frozen
    record-to-skill similarity cos(e(B_t), e(k)) as an optional anchor feature).
    """

    def __init__(self, d, hidden=512, use_anchor=True):
        super().__init__()
        self.use_anchor = use_anchor
        n_in = 3 * d + (1 if use_anchor else 0)
        self.net = nn.Sequential(nn.LayerNorm(n_in), nn.Linear(n_in, hidden), nn.GELU(),
                                 nn.Dropout(0.1), nn.Linear(hidden, 1))

    def forward(self, q, b, E):
        n = E.shape[0]
        Q = q.expand(n, -1)
        parts = [Q * E, (Q - E).abs(), E]
        if self.use_anchor:
            parts.append(E @ b.reshape(-1, 1))
        return self.net(torch.cat(parts, -1)).reshape(1, n)


def skill_logits(nav, head, head_kind, z, b, E):
    """Logits over the skill pool, shape (1, |K|).  z: (1, d_z) slot-mean, b: (1, d_txt)."""
    q = nav(z, b)
    if head_kind == "cross":
        return head(q, b[0], E) / TEMPERATURE
    return (q @ E.T) / TEMPERATURE


class NavController:
    """Serving-side f_theta + g_phi with per-episode recurrent state.

    query() folds this step's new text into the episode's Z_t (exactly as in training) and
    returns q_t; latent() returns the current Z_t (m x d_z) for the executor-side prefix
    without advancing the state.
    """

    def __init__(self, model_dir, enc_dir, nav_ckpt, device="cuda:0"):
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.dev = torch.device(device)
        self.tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16).to(self.dev)
        for p in base.parameters():
            p.requires_grad_(False)
        self.enc = PeftModel.from_pretrained(base, enc_dir).eval()
        self.emb = self.enc.get_base_model().get_input_embeddings().weight
        self.trunk = self.enc.base_model.model.model

        blob = torch.load(nav_ckpt, map_location=self.dev, weights_only=True)
        d = self.emb.shape[1]
        self.slots = nn.Parameter(blob["slots"].to(self.dev).float())
        self.init_state = nn.Parameter(blob["init_state"].to(self.dev).float())
        self.proj = nn.Linear(d, d).to(self.dev).float()
        self.proj.load_state_dict(blob["proj"])
        self.proj.eval()
        self.nav = Navigator(blob["d_z"], blob["d_txt"], blob["d_txt"]).to(self.dev)
        self.nav.load_state_dict(blob["nav"])
        self.nav.eval()
        self.val_r1 = float(blob.get("val_r1", -1))
        # the head kind rides inside the checkpoint so a scorer is never served as a pointer
        self.head_kind = blob.get("head", "pointer") or "pointer"
        self.head = None
        if self.head_kind == "cross":
            self.head = CrossScorer(blob["d_txt"],
                                    use_anchor=not blob.get("no_anchor_feat", False)).to(self.dev)
            self.head.load_state_dict(blob["head_state"])
            self.head.eval()
        elif self.head_kind != "pointer":
            raise ValueError("head %r is not part of this reference package" % self.head_kind)
        self.state = {}                      # episode id -> (Z_t, n_steps_folded)
        self.lock = threading.Lock()

    @torch.no_grad()
    def latent(self, fp):
        """Current Z_t of the episode, shape (m, d_z); None before the first fold."""
        with self.lock:
            st = self.state.get(fp)
        return None if st is None else st[0]

    @torch.no_grad()
    def query(self, fp, new_blocks, bt_vec, return_latent=False):
        """Fold only unseen completed transitions; retries do not advance state twice."""
        with self.lock:
            z, seen = self.state.get(fp, (self.init_state, []))
            hashes = [hashlib.sha256(block.encode()).hexdigest() for block in new_blocks]
            if hashes[:len(seen)] != seen:
                raise ValueError("episode history changed; provide a new episode key")
            if fp not in self.state and len(self.state) >= 512:
                raise ValueError("episode capacity exceeded; reset completed episodes")
            for block in new_blocks[len(seen):]:
                ids = self.tok(block, add_special_tokens=False)["input_ids"][-DELTA_MAX:]
                z = fold(self.emb, self.trunk, self.slots, self.proj, z, ids, self.dev)
            self.state[fp] = (z, hashes)
            t = torch.as_tensor(bt_vec, dtype=torch.float32, device=self.dev).view(1, -1)
            q = self.nav(z.mean(0, keepdim=True), t)
        vector = q[0].detach().float().cpu().numpy().astype(np.float32)
        return (vector, z) if return_latent else vector

    @torch.no_grad()
    def head_scores(self, pvec, bt_vec, card_emb):
        """Per-skill learned scores for the cross head (None for the pointer head).

        Takes the already-computed q_t; it must not fold again, or Z would run one step ahead.
        """
        if self.head is None:
            return None
        q = torch.as_tensor(pvec, dtype=torch.float32, device=self.dev).view(1, -1)
        b = torch.as_tensor(bt_vec, dtype=torch.float32, device=self.dev).view(-1)
        E = torch.as_tensor(np.asarray(card_emb), dtype=torch.float32, device=self.dev)
        return self.head(q, b, E)[0].float().cpu().numpy()
