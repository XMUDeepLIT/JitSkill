# -*- coding: utf-8 -*-
"""Latent prefix."""
import base64
import io

import torch

PART_TYPE = "prompt_embeds"
SENTINEL = "<prompt_embeds>"   # the marker vLLM renders for a prompt_embeds part


def encode_latent(z):
    """(num_tokens, hidden_size) float tensor -> the base64 payload vLLM expects."""
    t = z.detach().to("cpu") if isinstance(z, torch.Tensor) else torch.as_tensor(z)
    if t.dim() == 3 and t.shape[0] == 1:
        t = t[0]
    if t.dim() != 2:
        raise ValueError("latent must be 2D (num_tokens, hidden_size); got %s" % (tuple(t.shape),))
    if not t.is_floating_point():
        raise ValueError("latent must be floating point; got %s" % t.dtype)
    buf = io.BytesIO()
    torch.save(t.contiguous(), buf)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def make_part(z):
    return {"type": PART_TYPE, "data": encode_latent(z)}


def attach(messages, z, role="user"):
    """Append the latent part to the LAST message of `role` (the same turn the skill block and
    B_t were appended to), leaving everything else untouched.  A string content is promoted to
    a single text part, so the text the executor reads is unchanged character for character."""
    out = [dict(m) for m in messages]
    idx = None
    for i in range(len(out) - 1, -1, -1):
        if out[i].get("role") == role:
            idx = i
            break
    if idx is None:
        raise ValueError("no %r message to attach the latent to" % role)
    c = out[idx].get("content")
    if isinstance(c, str):
        parts = [{"type": "text", "text": c}]
    elif isinstance(c, list):
        parts = [dict(p) for p in c]
    else:
        raise ValueError("unsupported content type %s" % type(c))
    parts.append(make_part(z))
    out[idx]["content"] = parts
    return out
