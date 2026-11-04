# -*- coding: utf-8 -*-
"""Check position parity."""
import argparse
import json

import torch

from .latent_prefix import SENTINEL, attach
from .train_projection import render

MESSAGES = [
    {"role": "system", "content": "You are an agent that writes Python to complete a task."},
    {"role": "user", "content": "Task: find the most-liked song across my playlists."},
    {"role": "assistant", "content": "print(apis.spotify.show_playlist_library())"},
    {"role": "user", "content": "Execution output:\n[{'playlist_id': 1, 'title': 'demo'}]"},
]


def server_prompt_tokens(url, model, messages):
    import requests
    r = requests.post(url + "/chat/completions", timeout=180, json={
        "model": model, "messages": messages, "max_tokens": 1, "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False}})
    if r.status_code != 200:
        raise SystemExit("server refused the probe (HTTP %d): %s" % (r.status_code, r.text[:400]))
    return r.json()["usage"]["prompt_tokens"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--n_latent", type=int, default=16)
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=True)
    tok.add_special_tokens({"additional_special_tokens": [SENTINEL]})
    ph_ids = tok(SENTINEL, add_special_tokens=False)["input_ids"]
    if len(ph_ids) != 1:
        raise SystemExit("%r encodes to %d tokens, not 1" % (SENTINEL, len(ph_ids)))
    ph = ph_ids[0]
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(a.tokenizer)
    hidden = getattr(getattr(cfg, "text_config", cfg), "hidden_size")
    fails = []

    local_plain = len(tok(tok.apply_chat_template(MESSAGES, tokenize=False, add_generation_prompt=True,
                                                enable_thinking=False), add_special_tokens=False)["input_ids"])
    srv_plain = server_prompt_tokens(a.base_url, a.model, MESSAGES)
    print("1 %s template parity: local %d, server %d"
          % ("ok  " if srv_plain == local_plain else "FAIL", local_plain, srv_plain))
    if srv_plain != local_plain:
        fails.append("template")

    p_ids = tok(render(tok, MESSAGES), add_special_tokens=False)["input_ids"]
    if p_ids.count(ph) != 1:
        raise SystemExit("expected exactly 1 marker in the rendered prompt")
    if len(p_ids) - 1 != local_plain:
        fails.append("marker")
        print("2 FAIL  the marker perturbed the surrounding tokens")
    expect = local_plain + a.n_latent
    z = torch.randn(a.n_latent, hidden, dtype=torch.bfloat16)
    srv_lat = server_prompt_tokens(a.base_url, a.model, attach(MESSAGES, z))
    print("2 %s expansion parity: expected %d, server %d (marker index %d)"
          % ("ok  " if srv_lat == expect else "FAIL", expect, srv_lat, p_ids.index(ph)))
    if srv_lat != expect:
        fails.append("expansion")
    print("TOKEN_COUNT_PARITY %s" % ("FAIL (%s)" % ", ".join(fails) if fails else "PASS"))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
