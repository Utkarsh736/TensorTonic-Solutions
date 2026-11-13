import torch
import torch.nn.functional as F

def _batch_norm(x, scale, shift, eps):
    mean = x.mean(dim=(0, 2, 3), keepdim=True)
    var = x.var(dim=(0, 2, 3), unbiased=False, keepdim=True)

    scale = scale.view(1, -1, 1, 1)
    shift = shift.view(1, -1, 1, 1)

    return scale * (x - mean) / torch.sqrt(var + eps) + shift

def policy_value_heads(
    features: torch.Tensor,
    policy_conv_weight: torch.Tensor,
    policy_scale: torch.Tensor,
    policy_shift: torch.Tensor,
    policy_linear_weight: torch.Tensor,
    policy_linear_bias: torch.Tensor,
    value_conv_weight: torch.Tensor,
    value_scale: torch.Tensor,
    value_shift: torch.Tensor,
    value_hidden_weight: torch.Tensor,
    value_hidden_bias: torch.Tensor,
    value_output_weight: torch.Tensor,
    value_output_bias: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns: float32 tensors (policy_logits, values), shaped (B, H*W+1) and (B,).
    """
    device = features.device
    f = features.to(dtype=torch.float32, device=device)

    w_pc = policy_conv_weight.to(dtype=torch.float32, device=device)
    s_p = policy_scale.to(dtype=torch.float32, device=device)
    b_p = policy_shift.to(dtype=torch.float32, device=device)
    w_p = policy_linear_weight.to(dtype=torch.float32, device=device)
    b_p_lin = policy_linear_bias.to(dtype=torch.float32, device=device)

    w_vc = value_conv_weight.to(dtype=torch.float32, device=device)
    s_v = value_scale.to(dtype=torch.float32, device=device)
    b_v = value_shift.to(dtype=torch.float32, device=device)
    w_h = value_hidden_weight.to(dtype=torch.float32, device=device)
    b_h = value_hidden_bias.to(dtype=torch.float32, device=device)
    w_v = value_output_weight.to(dtype=torch.float32, device=device)
    b_v_out = value_output_bias.to(dtype=torch.float32, device=device)

    eps = torch.as_tensor(epsilon, dtype=torch.float32, device=device)

    p_conv = F.conv2d(f, w_pc, stride=1, padding=0, bias=None)

    p = torch.relu(_batch_norm(p_conv, s_p, b_p, eps))

    p_flat = p.flatten(1)

    policy_logits = F.linear(p_flat, w_p, b_p_lin)

    v_conv = F.conv2d(f, w_vc, stride=1, padding=0, bias=None)

    v = torch.relu(_batch_norm(v_conv, s_v, b_v, eps))

    v_flat = v.flatten(1)

    v_hidden = torch.relu(F.linear(v_flat, w_h, b_h))

    v_out = torch.tanh(F.linear(v_hidden, w_v, b_v_out))
    values = v_out.squeeze(-1)

    return policy_logits, values
