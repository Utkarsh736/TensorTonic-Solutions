import torch

def normalize_branches(branches: torch.Tensor, norm_weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return branches * torch.rsqrt(branches.pow(2).mean(dim=-1, keepdim=True) + eps) * (1 + norm_weight)

def gated_residual_write(branches: torch.Tensor, norm_weight: torch.Tensor, write_weight: torch.Tensor, sublayer_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns a tuple of the updated branches shaped (batch, seq_len, branches, d) and the write scales shaped (batch, seq_len, branches).
    """
    nr = branches.shape[-2]

    r_hat = normalize_branches(branches, norm_weight)
    vec = r_hat.flatten(start_dim=-2)
    scales = 2.0 * torch.sigmoid((vec @ write_weight.T) / nr)

    updated = branches + scales.unsqueeze(-1) * sublayer_output.unsqueeze(-2)

    return updated, scales
