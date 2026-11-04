# -*- coding: utf-8 -*-
"""Latent state."""
import torch
import torch.nn as nn

N_LATENT = 16      # m
ENC_MAX = 4096     # encoder window (tokens)
DELTA_MAX = 1536   # tokens of the new block D_{t+1} read per update (most recent kept)

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]


class LatentEncoder(nn.Module):
    """f_theta: frozen trunk + trainable LoRA, slots, Z_0 and projection.

    """

    def __init__(self, model_dir, device, lora_r=32, lora_alpha=64, lora_dropout=0.05):
        super().__init__()
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.dev = torch.device(device)
        self.tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        # frozen trunk in bf16; trainable LoRA master weights kept in fp32 (mixed precision --
        # bf16 LoRA weights produced non-finite gradients)
        base = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16).to(self.dev)
        for p in base.parameters():
            p.requires_grad_(False)
        cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                         target_modules=LORA_TARGETS, task_type="CAUSAL_LM")
        self.enc = get_peft_model(base, cfg)
        for p in self.enc.parameters():
            if p.requires_grad:
                p.data = p.data.float()
        self.emb = self.enc.get_base_model().get_input_embeddings().weight
        self.trunk = self.enc.base_model.model.model
        d = self.emb.shape[1]
        # learnable slots: the query positions Z_{t+1} is read out of
        self.slots = nn.Parameter(torch.randn(N_LATENT, d, device=self.dev) * 0.02)
        # Z_0: small random rather than zeros (an all-zero row is degenerate for RMSNorm)
        self.init_state = nn.Parameter(torch.randn(N_LATENT, d, device=self.dev) * 0.02)
        self.proj = nn.Linear(d, d).to(self.dev)

    @property
    def d_z(self):
        return self.emb.shape[1]

    def step(self, z_prev, ids):
        """One recurrent update  Z_{t+1} = f_theta(Z_t, D_{t+1}).  Differentiable."""
        return fold(self.emb, self.trunk, self.slots, self.proj, z_prev, ids, self.dev)

    def save_trainable_checkpoint(self, path):
        """Save LoRA, slot, and recurrent-projection weights without the frozen trunk."""
        state = {name: param.detach().cpu() for name, param in self.named_parameters()
                 if param.requires_grad}
        torch.save({"trainable": state, "n_latent": N_LATENT, "d_z": self.d_z}, path)

    def load_trainable_checkpoint(self, path):
        """Restore a compatible compression-pretraining checkpoint before navigation training."""
        blob = torch.load(path, map_location=self.dev, weights_only=True)
        if blob.get("n_latent") != N_LATENT or blob.get("d_z") != self.d_z:
            raise ValueError("pretraining checkpoint does not match the encoder interface")
        source = blob["trainable"]
        restored = 0
        for name, param in self.named_parameters():
            if param.requires_grad and name in source:
                param.data.copy_(source[name].to(device=param.device, dtype=param.dtype))
                restored += 1
        if not restored:
            raise ValueError("pretraining checkpoint contains no trainable encoder weights")


def fold(emb, trunk, slots, proj, z_prev, ids, dev):
    """[Z_t ; E(D_{t+1}) ; slots] -> trunk -> proj(hidden at the m slot positions)."""
    e = (emb[torch.tensor(ids, device=dev)].float() if ids
         else torch.zeros(0, emb.shape[1], device=dev))
    parts = ([z_prev] if z_prev is not None else []) + [e, slots.float()]
    x = torch.cat(parts, 0).unsqueeze(0).to(emb.dtype)
    h = trunk(inputs_embeds=x, use_cache=False).last_hidden_state[0]
    return proj(h[-N_LATENT:].float())


def episode_forward(encoder, step_token_ids, detach=True):
    """Walk one training trajectory, yielding (t, Z_t) after folding step t's block.

    Every step folds its own block, so Z_t is informative from the first step.
    `detach=True` keeps the recurrent forward dependency but truncates gradients across steps
    (Appendix: "gradients are truncated across steps"), so training memory does not grow with
    trajectory length.
    """
    z = encoder.init_state.float()
    for i, ids in enumerate(step_token_ids):
        delta = ids[-DELTA_MAX:]
        if delta:
            z = encoder.step(z, delta[-ENC_MAX:])
        yield i, z
        z = z.detach() if detach else z
