# -*- coding: utf-8 -*-
"""Inject proxy."""
import argparse
import json
import os
import sys
import threading
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jitskill.explicit_record import (MEM_HEADER, ExplicitRecord, episode_fp,  # noqa: E402
                                         last_turns, navigation_record, task_instruction,
                                         history_transitions, delta_text, text_content)
from jitskill.fusion import SkillIndex, render, step_text  # noqa: E402
from jitskill.library import load_pool  # noqa: E402

# Header of the skill block: injected skills are reference material, not executed work.
from jitskill.prompts import SKILL_HEADER as HEADER
TERMINAL = "supervisor.complete_task"


def parse_model(name):
    info = {"k": 3, "bud": 1200, "lk": ""}
    for part in (name or "").split("::"):
        if "=" not in part:
            continue
        key, _, val = part.partition("=")
        if key == "lk":
            info["lk"] = val
        elif key in ("k", "bud"):
            info[key] = int(val)
    return info


class InjectionService:
    def __init__(self, a):
        self.a = a
        self.pool = load_pool(a.skills, docs_dir=a.docs_dir or None)
        self.index = SkillIndex(self.pool, emb_model=a.emb_model, device=a.emb_device,
                                cache=a.emb_cache or None)
        self.upstream = a.upstream.rstrip("/") + "/v1/chat/completions"
        self._tl = threading.local()       # per-request Z_t handed from block_for to complete
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=True)
        # f_theta + g_phi always run on their own Qwen3-8B base, independent of the executor
        from jitskill.navigator import NavController
        self.ctrl = NavController(a.nav_base, a.enc_dir, a.nav_ckpt, device=a.nav_device)
        print("[proxy] navigator loaded (val R@1 %.3f) head=%s on %s"
              % (self.ctrl.val_r1, self.ctrl.head_kind, a.nav_device), flush=True)
        self.projector = None
        if a.exec_latent:
            if not a.projector:
                raise ValueError("--projector is required for projected latent-state execution")
            from jitskill.executor.projection import StateProjector
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(a.tokenizer)
            h = getattr(getattr(cfg, "text_config", cfg), "hidden_size", None)
            self.projector = StateProjector.load(a.projector, a.nav_device).eval()
            if (self.projector.linear.in_features != self.ctrl.emb.shape[1]
                    or self.projector.linear.out_features != h):
                raise ValueError("projector dimensions do not match encoder and executor")

    def ntok(self, s):
        return len(self.tok(s, add_special_tokens=False)["input_ids"])

    # ------------------------------------------------------------------ skill acquisition
    def block_for(self, info, msgs):
        """Text injected at decision point t, plus the selected skill ids."""
        if not isinstance(info.get("lk"), str) or not info["lk"].strip():
            raise ValueError("model must include a nonempty lk=<unique-episode-id>")
        fp = episode_fp(msgs, info["lk"])
        goal = task_instruction(msgs)                       # u
        say, obs = last_turns(msgs)
        transitions = history_transitions(msgs)
        # Rebuild B_t from raw preceding history, making repeated requests idempotent.
        record = ExplicitRecord(goal)
        for action, feedback in transitions:
            record.observe(action, feedback)
        bt = navigation_record(record)
        bt_vec = self.index.embed(bt)
        blocks = [delta_text(action, feedback) for action, feedback in transitions]
        pvec, z = self.ctrl.query(fp, blocks, bt_vec, return_latent=True)
        self._tl.latent = z  # same snapshot used for navigation and corpus construction

        # --- fusion with lexical relevance of the current step, then K_t under C_K ----------
        stxt = step_text(say, obs)
        lex_text = bt if self.a.lex == "bt" else stxt
        dense = self.ctrl.head_scores(pvec, bt_vec, self.index.emb)   # None for pointer head
        sc = self.index.score_fused(pvec, lex_text, stxt, alpha=self.a.alpha,
                                    layer="procedural", dense_scores=dense)
        hits = self.index.select(sc, k=info["k"], budget_tokens=info["bud"],
                                 count_tokens=self.ntok)

        # --- skill-use side: K_t and B_t as text --------------------------------------------
        block = render(hits, HEADER)
        if self.a.inject_record:
            block = MEM_HEADER + "\n" + record.render() + "\n\n" + block
        return block, [h["card"]["id"] for h in hits]

    @staticmethod
    def no_terminal(block):
        """Never hand the executor an episode-ending call inside reference material."""
        if not block:
            return block
        return "\n".join(ln for ln in block.splitlines() if TERMINAL not in ln)

    def rewrite(self, msgs, block):
        """Append the block to the latest user turn (immediately before the executor's turn)."""
        if not block:
            return msgs
        out = [dict(m) for m in msgs]
        for i in range(len(out) - 1, -1, -1):
            if out[i].get("role") == self.a.injection_role:
                out[i]["content"] = text_content(out[i]) + "\n\n" + block
                return out
        if self.a.injection_role == "tool":
            raise ValueError("SkillsBench tool injection requires an existing tool result")
        if self.a.injection_role == "system":
            out.insert(0, {"role": "system", "content": block})
        else:
            out.append({"role": self.a.injection_role, "content": block})
        return out

    # ------------------------------------------------------------------ skill use (executor)
    def complete(self, body):
        if body.get("stream"):
            return 400, {"error": {"message": "streaming is not supported"}}
        try:
            info = parse_model(body.get("model", ""))
        except (TypeError, ValueError):
            return 400, {"error": {"message": "invalid proxy model control fields"}}
        if not info["lk"].strip():
            return 400, {"error": {"message": "model must include a nonempty lk=<unique-episode-id>"}}
        msgs = body.get("messages") or []
        if self.a.injection_role == "tool" and not any(m.get("role") == "tool" for m in msgs):
            return 400, {"error": {"message": "SkillsBench adapter must provide an injectable tool result"}}
        if any(MEM_HEADER in text_content(m) for m in msgs):
            return 400, {"error": {"message": "requests must contain raw pre-injection history"}}
        self._tl.latent = None
        block, _ids = self.block_for(info, msgs)
        block = self.no_terminal(block)
        fwd = {k: v for k, v in body.items() if k != "stream"}
        fwd["messages"] = self.rewrite(msgs, block)          # B_t + K_t as text
        fwd["model"] = self.a.upstream_model
        z = getattr(self._tl, "latent", None)
        if self.a.exec_latent and z is not None:
            from jitskill.executor.latent_prefix import attach
            # Z_t as a 16-position prompt_embeds part on the same user turn
            import torch
            with torch.no_grad():
                projected = self.projector(z)
            fwd["messages"] = attach(fwd["messages"], projected.to(torch.bfloat16),
                                     role=self.a.injection_role)
        if self.a.no_think:
            kw = dict(fwd.get("chat_template_kwargs") or {})
            kw["enable_thinking"] = False
            fwd["chat_template_kwargs"] = kw
        req = urllib.request.Request(
            self.upstream, data=json.dumps(fwd).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer " + os.environ.get("JITSKILL_UPSTREAM_KEY",
                                                                 "EMPTY")})
        try:
            with urllib.request.urlopen(req, timeout=self.a.timeout) as r:
                return 200, json.loads(r.read())
        except urllib.error.HTTPError as e:
            payload = e.read()
            try:
                return e.code, json.loads(payload)
            except Exception:
                return e.code, {"error": {"message": payload.decode("utf-8", "replace"),
                                          "code": e.code}}


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--upstream", required=True, help="executor engine base URL")
    ap.add_argument("--upstream_model", required=True, help="served model name of the executor")
    ap.add_argument("--tokenizer", required=True, help="executor model dir (tokenizer, config)")
    ap.add_argument("--skills", required=True, help="skill library jsonl")
    ap.add_argument("--docs_dir", default="", help="optional AppWorld api_docs dir")
    ap.add_argument("--emb_model", default=os.environ.get(
        "JITSKILL_EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    ap.add_argument("--emb_device", default="cpu")
    ap.add_argument("--emb_cache", default="")
    ap.add_argument("--nav_base", required=True, help="encoder base (Qwen3-8B)")
    ap.add_argument("--nav_ckpt", required=True, help="<prefix>_nav.pt from train_navigator")
    ap.add_argument("--enc_dir", required=True, help="<prefix>_enc/ from train_navigator")
    ap.add_argument("--nav_device", default="cuda:0")
    ap.add_argument("--alpha", type=float, required=True,
                    help="weight of the learned view vs the frozen step-text view")
    ap.add_argument("--lex", choices=("step", "bt"), default="step",
                    help="text of the lexical (BM25) half: current step, or the B_t summary")
    ap.add_argument("--inject_record", type=int, default=1,
                    help="1: inject B_t as text to the executor (reported configuration)")
    ap.add_argument("--projector", default="", help="trained P_psi checkpoint")
    ap.add_argument("--injection_role", choices=["user", "tool", "system"], default="user",
                    help="ALFWorld/AppWorld: user; SkillsBench: tool; tau2-Bench: system")
    ap.add_argument("--exec_latent", type=int, default=1,
                    help="1: attach P_psi(Z_t) as prompt_embeds to the frozen executor; "
                         "0: text-only executor")
    ap.add_argument("--no_think", type=int, default=1,
                    help="force enable_thinking=false on the executor chat template")
    ap.add_argument("--timeout", type=float, default=1200.0)
    return ap


def main():
    a = build_argparser().parse_args()
    svc = InjectionService(a)
    print("[proxy] pool=%d cards, %d skills; upstream=%s"
          % (len(svc.pool), sum(1 for c in svc.pool if c["layer"] == "procedural"),
             a.upstream), flush=True)

    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_):
            pass

        def _send(self, code, obj):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.rstrip("/").endswith("/models"):
                self._send(200, {"object": "list", "data": [{"id": "jitskill",
                                                             "object": "model"}]})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except Exception as e:
                return self._send(400, {"error": {"message": str(e)}})
            try:
                code, resp = svc.complete(body)
            except Exception as e:
                return self._send(500, {"error": {"message": "%s: %s" % (type(e).__name__, e)}})
            self._send(code, resp)

    ThreadingHTTPServer((a.host, a.port), H).serve_forever()


if __name__ == "__main__":
    main()
