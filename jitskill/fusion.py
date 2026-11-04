# -*- coding: utf-8 -*-
"""Fusion."""
import os
import hashlib
import json
import re

import numpy as np

TOKEN_RE = re.compile(r"[a-z0-9_]+")
EMB_MODEL = os.environ.get("JITSKILL_EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RRF_C = 60.0


def tokenize(s):
    return TOKEN_RE.findall((s or "").lower())


def step_text(thought, observation):
    """The current step's text: latest thought + tail of the observation (lexical relevance)."""
    return "\n".join(x for x in [(thought or "").strip(),
                                 (observation or "").strip()[-1200:]] if x)


def body(card):
    """Text injected for a skill (a compact `inject` pointer if the card carries one)."""
    return card.get("inject") or card["text"]


def render(hits, header):
    if not hits:
        return ""
    return header + "\n\n" + "\n\n".join(body(h["card"]) for h in hits)


class SkillIndex:
    """BM25 + frozen dense embeddings over the skill pool.

    """

    def __init__(self, pool, emb_model=EMB_MODEL, device="cpu", cache=None):
        from rank_bm25 import BM25Okapi
        from sentence_transformers import SentenceTransformer
        self.pool = pool
        self.texts = [c["text"] for c in pool]
        self.bm25 = BM25Okapi([tokenize(t) for t in self.texts])
        self.model = SentenceTransformer(emb_model, device=device)
        key = hashlib.sha256(json.dumps([emb_model, self.texts], ensure_ascii=False).encode()).hexdigest()
        cached = np.load(cache, allow_pickle=False) if cache and os.path.exists(cache) else None
        try:
            valid = cached is not None and "key" in cached and str(cached["key"]) == key
            if valid:
                self.emb = cached["emb"].copy()
            else:
                self.emb = self.model.encode(self.texts, batch_size=32, convert_to_numpy=True,
                                             normalize_embeddings=True, show_progress_bar=False)
                if cache:
                    np.savez(cache, emb=self.emb, key=key)
        finally:
            if cached is not None:
                cached.close()

    def embed(self, text):
        return self.model.encode([text], convert_to_numpy=True, normalize_embeddings=True,
                                 show_progress_bar=False)[0]

    @staticmethod
    def _ranks(scores):
        order = np.argsort(-scores)
        r = np.empty(len(scores), dtype=np.int64)
        r[order] = np.arange(len(scores))
        return r

    def score_fused(self, qvec, lex_text, dense_text, alpha=1.0, c=RRF_C, layer="procedural",
                    dense_scores=None):
        """S(k) as in the module docstring.  `dense_scores` (cross head) replaces cos(q_t, e(k))."""
        lex = np.asarray(self.bm25.get_scores(tokenize(lex_text)), dtype=np.float64)
        fused = 1.0 / (c + self._ranks(lex))
        if dense_scores is not None:
            dv = np.asarray(dense_scores, dtype=np.float64)
            if dv.shape[0] != len(self.pool):
                raise ValueError("dense_scores covers %d skills, pool has %d"
                                 % (dv.shape[0], len(self.pool)))
        else:
            q = np.asarray(qvec, dtype=np.float32)
            q = q / (np.linalg.norm(q) + 1e-8)
            dv = self.emb @ q
        fused = fused + alpha * (1.0 / (c + self._ranks(dv)))
        if alpha < 1.0:
            t = self.embed(dense_text)
            fused = fused + (1.0 - alpha) * (1.0 / (c + self._ranks(self.emb @ t)))
        mask = np.full(len(self.pool), -np.inf)
        for i, card in enumerate(self.pool):
            if layer is None or card["layer"] == layer:
                mask[i] = 0.0
        return fused + mask

    def select(self, scores, k=3, budget_tokens=1200, count_tokens=None):
        """K_t: top-k by fused score, packed into C_K (a skill that does not fit is skipped).

        """
        if k < 0 or (budget_tokens is not None and budget_tokens < 0):
            raise ValueError("k and budget must be non-negative")
        if budget_tokens is not None and count_tokens is None:
            raise ValueError("a tokenizer is required for a token budget")
        order = np.argsort(-scores)
        out, used = [], 0
        for i in order:
            if len(out) >= k:
                break
            if not np.isfinite(scores[i]):
                continue
            card = self.pool[int(i)]
            if budget_tokens is not None and count_tokens is not None:
                n = count_tokens(body(card))
                if used + n > budget_tokens:
                    continue
                used += n
            out.append({"card": card, "score": float(scores[i])})
        return out
