import torch
import torch.nn.functional as F

def _batch_norm(x, scale, shift, eps):
    mean = x.mean(dim=(0, 2, 3), keepdim=True)         
    var = x.var(dim=(0, 2, 3), unbiased=False, keepdim=True)

    scale = scale.view(1, -1, 1, 1)
    shift = shift.view(1, -1, 1, 1)

    return scale * (x - mean) / torch.sqrt(var + eps) + shift

def alpha_go_residual_block(
    x: torch.Tensor,
    conv1_weight: torch.Tensor,
    scale1: torch.Tensor,
    shift1: torch.Tensor,
    conv2_weight: torch.Tensor,
    scale2: torch.Tensor,
    shift2: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """
    Returns: the float32 residual output, with the same shape and device as x.
    """
    device = x.device

    x = x.to(dtype=torch.float32, device=device)

    w1 = conv1_weight.to(dtype=torch.float32, device=device)
    w2 = conv2_weight.to(dtype=torch.float32, device=device)

    s1 = scale1.to(dtype=torch.float32, device=device)
    b1 = shift1.to(dtype=torch.float32, device=device)
    s2 = scale2.to(dtype=torch.float32, device=device)
    b2 = shift2.to(dtype=torch.float32, device=device)

    eps = torch.as_tensor(epsilon, dtype=torch.float32, device=device)

    conv1 = F.conv2d(x, w1, stride=1, padding=1, bias=None)
    h = torch.relu(_batch_norm(conv1, s1, b1, eps))

    conv2 = F.conv2d(h, w2, stride=1, padding=1, bias=None)
    branch = _batch_norm(conv2, s2, b2, eps)

    y = torch.relu(x + branch)

    return y
