import torch

def rmsnorm(x: torch.Tensor, g: torch.Tensor, epsilon: float) -> torch.Tensor:
    """
    Returns the RMS-normalized tensor with the same shape and dtype as x.
    """
    x_float = x.float()
    g_float = g.float()
    mean_sq = torch.mean(x.pow(2), dim=-1, keepdim=True)

    return (x_float*g_float/torch.sqrt(mean_sq+epsilon)).to(x.dtype)