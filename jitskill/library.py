# -*- coding: utf-8 -*-
"""Library."""
import json
import os

TERMINAL = "supervisor.complete_task"


def _fmt_params(params):
    out = []
    for p in params or []:
        if not isinstance(p, dict):
            continue
        req = "required" if p.get("required") else "optional"
        dflt = "" if p.get("default") in (None, "") else ", default=%s" % (p["default"],)
        out.append("  - %s (%s, %s%s): %s" % (p.get("name"), p.get("type", "?"), req, dflt,
                                              (p.get("description") or "").strip()))
    return "\n".join(out)


def load_api_cards(docs_dir, fmt="standard"):
    """One declarative card per official AppWorld API (reads <docs_dir>/<fmt>/<app>.json)."""
    d = os.path.join(docs_dir, fmt)
    cards = []
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".json") or fn == "api_docs.json":
            continue
        app = fn[:-5]
        obj = json.load(open(os.path.join(d, fn), encoding="utf-8"))
        items = obj.items() if isinstance(obj, dict) else (
            (e.get("api_name"), e) for e in obj if isinstance(e, dict))
        for name, spec in items:
            if not name:
                continue
            spec = spec or {}
            lines = ["[API] %s.%s  %s %s" % (app, name, spec.get("method", ""),
                                             spec.get("path", "")),
                     (spec.get("description") or "").strip()]
            ps = _fmt_params(spec.get("parameters"))
            if ps:
                lines.append("PARAMS:\n" + ps)
            cards.append({"id": "api::%s.%s" % (app, name), "layer": "declarative",
                          "app": app, "apis": ["%s.%s" % (app, name)],
                          "text": "\n".join(x for x in lines if x)})
    return cards


def _strip_terminal(text):
    """Injected material is reference only: drop any line that would end the episode."""
    return "\n".join(ln for ln in str(text).splitlines() if TERMINAL not in ln).strip()


def _skill_sections(sk):
    out = [("WHEN", sk.get("trigger", ""))]
    pre = sk.get("preconditions") or []
    if pre:
        out.append(("NEEDS", "\n".join("- " + str(p) for p in pre)))
    for st in sk.get("procedure") or []:
        if isinstance(st, dict):
            out.append(("STEP %s" % st.get("step", "?"),
                        "%s\n%s" % (st.get("what", ""), st.get("code", ""))))
    if sk.get("function"):
        out.append(("FUNCTION", sk["function"]))
    fh = sk.get("failure_handling") or []
    if fh:
        out.append(("IF IT FAILS", "\n".join("- " + str(f) for f in fh)))
    ex = sk.get("examples") or []
    if ex:
        out.append(("EXAMPLE", "\n".join(str(e) for e in ex)))
    cleaned = [(lab, _strip_terminal(txt)) for lab, txt in out]
    return [(lab, txt) for lab, txt in cleaned if txt]


def load_skill_cards(path):
    """Procedural cards.  A jsonl row is either an already-rendered card ({"id","text",...})
    or a structured skill record ({"skill_id","name","trigger","procedure",...})."""
    cards = []
    for line in open(path, encoding="utf-8"):
        sk = json.loads(line)
        if "id" in sk and "text" in sk:
            if not sk["id"].startswith("frag::"):
                sk = dict(sk)
                sk.setdefault("layer", "procedural")
                sk["text"] = _strip_terminal(sk["text"])
                if sk.get("inject"):
                    sk["inject"] = _strip_terminal(sk["inject"])
                cards.append(sk)
            continue
        head = "[SKILL] %s  (%s)" % (sk.get("name"), sk.get("app"))
        text = "\n".join("%s: %s" % (lab, txt) for lab, txt in _skill_sections(sk))
        cards.append({"id": "skill::%s" % sk["skill_id"], "layer": "procedural",
                      "app": sk.get("app"), "apis": sk.get("apis", []),
                      "scenario": sk.get("scenario"), "text": head + "\n" + text})
    return cards


def load_pool(skills_path, docs_dir=None):
    """The scoring pool: [API cards (if docs_dir)] + skill cards (skills last, in order)."""
    cards = load_skill_cards(skills_path)
    if docs_dir:
        cards = load_api_cards(docs_dir) + cards
    for i, c in enumerate(cards):
        c["idx"] = i
    return cards
