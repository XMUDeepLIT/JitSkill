"""P_psi: encoder slots -> frozen executor input embeddings (Eq. latent-state-projection)."""
import torch
from torch import nn

class StateProjector(nn.Module):
    def __init__(self, d_z, d_exec):
        super().__init__()
        self.linear = nn.Linear(d_z, d_exec)

    def forward(self, z):
        if z.shape[-1] != self.linear.in_features:
            raise ValueError("encoder latent dimension does not match projector")
        return self.linear(z.float())

    def save(self, path):
        torch.save({"d_z": self.linear.in_features, "d_exec": self.linear.out_features,
                    "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path, device="cpu"):
        blob = torch.load(path, map_location=device, weights_only=True)
        model = cls(blob["d_z"], blob["d_exec"]).to(device)
        model.load_state_dict(blob["state_dict"])
        return model
