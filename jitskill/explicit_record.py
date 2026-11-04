"""Bounded, evidence-only task progress shared by navigation and execution."""
import hashlib
import json
import re
from .prompts import MEM_HEADER

TASK_MARK = "now generate code to solve the actual task"
API_CALL_RE = re.compile(r"\bapis\.([a-z_]+)\.([a-z_0-9]+)\s*\(")
ERROR_RE = re.compile(r"Exception|Error|Traceback|failed|Nothing happened|invalid action", re.I)


def text_content(message):
    c = message.get("content") or ""
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "\n".join(p.get("text", "") for p in c if p.get("type") == "text")
    raise ValueError("message content must be text or a list of content parts")


def task_instruction(msgs):
    for m in reversed(msgs):
        c = text_content(m)
        if m.get("role") == "user" and TASK_MARK in c:
            return c.split(TASK_MARK, 1)[1].lstrip(": \n").strip()[:400]
    # Generic environments provide the task in the first user observation.
    return next((text_content(m)[:400] for m in msgs if m.get("role") == "user"), "")


def episode_fp(msgs, run_key=""):
    return hashlib.sha256((run_key + "||" + task_instruction(msgs)).encode()).hexdigest()[:24]


def actual_messages(msgs):
    """Exclude worked demonstrations from history when AppWorld supplies its task marker."""
    start = 0
    for i, m in enumerate(msgs):
        if m.get("role") == "user" and TASK_MARK in text_content(m):
            start = i
    return msgs[start:]


def history_transitions(msgs):
    """Completed action-feedback blocks only, preceding the next assistant decision."""
    pending, feedback, out = None, [], []
    for m in actual_messages(msgs):
        role, content = m.get("role"), text_content(m)
        if role == "assistant":
            if m.get("tool_calls"):
                content += "\nTOOL CALLS:\n" + json.dumps(m["tool_calls"], ensure_ascii=False)
            if pending is not None and feedback:
                out.append((pending, "\n".join(feedback)))
            pending, feedback = content, []
        elif role in ("user", "tool") and pending is not None:
            feedback.append(content)
    if pending is not None and feedback:
        out.append((pending, "\n".join(feedback)))
    return out


def action_text(response):
    match = re.search(r"^Action:\s*(.+)$", response, re.M)
    if match:
        return match.group(1).strip()
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", response, re.S)
    return "\n".join(blocks).strip() if blocks else response.strip()


def delta_text(action, observation):
    # Includes executed action/code as well as thought and feedback. Same at train/serve time.
    return "ACTION AND CONTEXT:\n" + action + "\nFEEDBACK:\n" + observation


def last_turns(msgs):
    blocks = history_transitions(msgs)
    if blocks:
        return blocks[-1]
    return "", next((text_content(m) for m in reversed(msgs)
                      if m.get("role") in ("user", "tool")), "")


class ExplicitRecord:
    """Paper prompt: goal <=400 characters; six actions and six failures, feedback <=90.

    Operations are observed attempts, never inferred successes. Result snippets retain key
    facts. Optional structured progress facts must be supplied by the environment adapter,
    with direct observation evidence; no LLM decomposition or verification is performed.
    """
    def __init__(self, goal):
        self.goal = goal[:400]
        self.actions, self.failures = [], []
        self.facts, self.subtasks = {}, {}
        self.total = 0

    def observe(self, response, observation, *, facts=None, subtasks=None):
        action = action_text(response)
        if action:
            entry = (action, " ".join(observation.split())[:90])
            self.actions.append(entry)
            self.total += 1
            self.actions = self.actions[-6:]
            if ERROR_RE.search(observation):
                self.failures.append(entry)
                self.failures = self.failures[-6:]
        for collection, updates in ((self.facts, facts), (self.subtasks, subtasks)):
            for key, update in (updates or {}).items():
                evidence = update.get("evidence", "")
                if not evidence or evidence not in observation:
                    raise ValueError("progress updates require direct observation evidence")
                collection.pop(key, None)
                collection[str(key)] = (str(update["value"])[:90], evidence[:90])
                while len(collection) > 6:
                    collection.pop(next(iter(collection)))

    def render(self, cap=6):
        if cap < 1:
            raise ValueError("cap must be positive")
        lines = ["GOAL: " + self.goal]
        if self.actions:
            lines.append("ACTIONS DONE (%d so far, latest last):" % self.total)
            lines.extend("- %s -> %s" % x for x in self.actions[-cap:])
        if self.failures:
            lines.append("FAILED ACTIONS:")
            lines.extend("- %s -> %s" % x for x in self.failures[-cap:])
        for label, entries in (("OBSERVED FACTS", self.facts), ("SUBTASK PROGRESS", self.subtasks)):
            if entries:
                lines.append(label + ":")
                lines.extend("- %s: %s [evidence: %s]" % (k, *v)
                             for k, v in list(entries.items())[-cap:])
        return "\n".join(lines)


def navigation_record(record, say="", obs=""):
    # The same B_t is embedded by the navigator and injected into the executor.
    return record.render()
