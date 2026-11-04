# -*- coding: utf-8 -*-
"""Objectives."""
import torch
import torch.nn.functional as F


def nav_target(gold, n_pool, gold_off=0, device="cpu"):
    """y_{i,t}: uniform mass over the gold skills (indices into the procedural cards)."""
    gold = sorted(set(gold))
    if not gold or any(k < 0 or gold_off + k >= n_pool for k in gold):
        raise ValueError("gold skill indices must be nonempty and inside the pool")
    tgt = torch.zeros(1, n_pool, device=device)
    for k in gold:
        tgt[0, gold_off + k] = 1.0 / len(gold)
    return tgt


def l_nav(logits, gold, gold_off=0):
    """Soft-label cross-entropy for one decision point.  logits: (1, |pool|)."""
    tgt = nav_target(gold, logits.shape[-1], gold_off, logits.device)
    return -(tgt * torch.log_softmax(logits, -1)).sum()


def l_exec(target_logits, target_ids):
    """Executor SFT loss: CE over the target (next action) tokens only.

    target_logits: logits at the len(target)+1 last positions, as returned with
    logits_to_keep=len(target)+1; the last position is dropped (it predicts past EOS).
    """
    return F.cross_entropy(target_logits[:-1].float(), target_ids)


def l_reconstruction(logits, target_ids):
    """Mean teacher-forced target-token CE (Eq. history-reconstruction)."""
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), target_ids.reshape(-1))


def l_alignment(teacher_logits, student_logits):
    """Normalized JSD of final-position predictions; teacher receives no gradients."""
    import math
    teacher = torch.softmax(teacher_logits.detach().float(), -1)
    student = torch.softmax(student_logits.float(), -1)
    mean = (teacher + student) / 2
    eps = torch.finfo(mean.dtype).tiny
    def kl(p):
        return (p * (p.clamp_min(eps).log() - mean.clamp_min(eps).log())).sum(-1)
    return ((kl(teacher) + kl(student)) / (2 * math.log(2))).mean()


def l_contrast(z, positive, negatives, temperature=0.07):
    """Slot mean versus normalized, frozen, centered-history anchors (Eq. representation-contrast).

    Callers must center history hidden representations before passing positive/negative anchors.
    """
    v = F.normalize(z.float().mean(-2), dim=-1)
    positive = F.normalize(positive.detach().float(), dim=-1)
    negatives = F.normalize(negatives.detach().float(), dim=-1)
    pos = (v * positive).sum(-1, keepdim=True) / temperature
    neg = v @ negatives.T / temperature
    return (-pos.squeeze(-1) + torch.logsumexp(torch.cat([pos, neg], -1), -1)).mean()


def l_compression(rec, align, disc, *, weight=1.0, lambda_rec=1.0,
                  lambda_probe=0.0, probe=0.0, lambda_align=1.0, lambda_disc=1.0):
    """Effective weighting in Eq. memory-training-full; not used by navigation training."""
    return weight * ((lambda_rec + lambda_probe * probe) * rec
                     + lambda_align * align + lambda_disc * disc)
