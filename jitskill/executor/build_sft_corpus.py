# -*- coding: utf-8 -*-
"""Build sft corpus."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from serving.inject_proxy import InjectionService, build_argparser  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="teacher experiment output dir")
    ap.add_argument("--successes", required=True, help="json list of solved task ids")
    ap.add_argument("--out", required=True)
    ap.add_argument("--latent_out", default="", help=".npy of shape (n_rows, m, d_z)")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--bud", type=int, default=1200)
    ap.add_argument("--max_ctx", type=int, default=15000)
    a, rest = ap.parse_known_args()
    # the service is configured with exactly the proxy's own flags (see serving/inject_proxy.py)
    svc_args = build_argparser().parse_args(
        rest + ["--upstream", "http://unused", "--upstream_model", "unused",
                "--exec_latent", "0"])
    svc = InjectionService(svc_args)
    if svc_args.injection_role != "user":
        raise ValueError("AppWorld corpus expects user-turn injection")
    lat = [] if a.latent_out else None
    succ = set(json.load(open(a.successes)))

    n_ex = n_skip = tot = 0
    tmp_out = a.out + ".part"
    with open(tmp_out, "w", encoding="utf-8") as fh:
        for tid in sorted(succ):
            lm = os.path.join(a.exp, "tasks", tid, "logs", "lm_calls.jsonl")
            if not os.path.exists(lm):
                continue
            for line in open(lm, encoding="utf-8"):
                rec = json.loads(line)
                msgs = (rec.get("input") or {}).get("messages") or []
                try:
                    target = rec["output"]["choices"][0]["message"]["content"] or ""
                except (KeyError, IndexError, TypeError):
                    continue
                if any("Progress notes for the current episode" in str(m.get("content", ""))
                       for m in msgs):
                    raise ValueError("corpus requires raw pre-injection requests")
                if not msgs or not target.strip():
                    continue
                svc._tl.latent = None
                info = {"k": a.k, "bud": a.bud, "lk": "sft_%s" % tid}
                block, ids = svc.block_for(info, msgs)
                prompt = svc.rewrite(msgs, svc.no_terminal(block))
                n = svc.ntok(json.dumps(prompt, ensure_ascii=False)) + svc.ntok(target)
                if n > a.max_ctx:
                    n_skip += 1
                    continue
                if lat is not None:
                    z = getattr(svc._tl, "latent", None)
                    if z is None:          # never write a row whose latent is missing
                        n_skip += 1
                        continue
                    lat.append(z.detach().to("cpu").float().numpy().astype("float16"))
                fh.write(json.dumps({"task_id": tid, "messages": prompt, "target": target,
                                     "retrieved": ids, "n_tokens": n},
                                    ensure_ascii=False) + "\n")
                n_ex += 1
                tot += n
    if lat is not None:
        import numpy as np
        arr = np.stack(lat) if lat else np.zeros((0, 0, 0), dtype="float16")
        if len(arr) != n_ex:
            raise SystemExit("latents %d != rows %d" % (len(arr), n_ex))
        np.save(a.latent_out, arr)
        print("[sft] latents %s -> %s" % (arr.shape, a.latent_out))
    os.replace(tmp_out, a.out)     # the jsonl appears only once its latents are on disk
    print("[sft] %d rows (skipped %d), mean %d tok -> %s"
          % (n_ex, n_skip, tot // max(1, n_ex), a.out))


if __name__ == "__main__":
    main()
