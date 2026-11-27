import torch
import torch.nn.functional as F

def gated_residual_read(branches: torch.Tensor, norm_weight: torch.Tensor, down_weight: torch.Tensor, up_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns a tuple of the sublayer input shaped (batch, seq_len, d) and the read gates shaped (batch, seq_len, branches, d).
    """
    eps = 1e-6
    nr = branches.shape[-2]

    scale = torch.rsqrt(branches.pow(2).mean(dim=-1, keepdim=True) + eps)
    r_hat = branches * scale * (1.0 + norm_weight)   
    
    vec = r_hat.flatten(start_dim=-2)                

    hidden = (vec @ down_weight.T) / nr              
    hidden = F.silu(hidden)
    gates = torch.sigmoid(hidden @ up_weight.T)       

    G = gates.unflatten(-1, (nr, r_hat.shape[-1]))    

    x = (G * r_hat).mean(dim=-2)     

    return x, G
