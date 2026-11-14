import torch
import torch.nn.functional as F

def _bn(z, scale, shift, eps):
    mean = z.mean(dim=(0, 2, 3), keepdim=True)
    var = z.var(dim=(0, 2, 3), unbiased=False, keepdim=True)
    scale = scale.view(1, -1, 1, 1)
    shift = shift.view(1, -1, 1, 1)
    return scale * (z - mean) / torch.sqrt(var + eps) + shift

def compact_policy_value_network(
    x: torch.Tensor,
    trunk_weight: torch.Tensor,
    trunk_scale: torch.Tensor,
    trunk_shift: torch.Tensor,
    residual_parameters: list[dict],
    head_parameters: dict,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns: float32 tensors (policy_logits, values), shaped (B, N*N+1) and (B,).
    """
    device = x.device

    def f32(t):
        return t.to(dtype=torch.float32, device=device)

    h = f32(x)

    eps = torch.as_tensor(epsilon, dtype=torch.float32, device=device)

    w_trunk = f32(trunk_weight)
    s_trunk = f32(trunk_scale)
    b_trunk = f32(trunk_shift)

    conv = F.conv2d(h, w_trunk, stride=1, padding=1, bias=None)
    h = torch.relu(_bn(conv, s_trunk, b_trunk, eps))

    for block in residual_parameters:
        w1 = f32(block["conv1_weight"])
        s1 = f32(block["scale1"])
        b1 = f32(block["shift1"])

        w2 = f32(block["conv2_weight"])
        s2 = f32(block["scale2"])
        b2 = f32(block["shift2"])

        shortcut = h

        r = F.conv2d(h, w1, stride=1, padding=1, bias=None)
        r = torch.relu(_bn(r, s1, b1, eps))

        r = F.conv2d(r, w2, stride=1, padding=1, bias=None)
        r = _bn(r, s2, b2, eps)

        h = torch.relu(shortcut + r)

    p_conv_w = f32(head_parameters["policy_conv_weight"])
    p_scale = f32(head_parameters["policy_scale"])
    p_shift = f32(head_parameters["policy_shift"])
    p_lin_w = f32(head_parameters["policy_linear_weight"])
    p_lin_b = f32(head_parameters["policy_linear_bias"])

    p = F.conv2d(h, p_conv_w, stride=1, padding=0, bias=None)
    p = torch.relu(_bn(p, p_scale, p_shift, eps))
    p = p.flatten(1)
    policy_logits = F.linear(p, p_lin_w, p_lin_b)

    v_conv_w = f32(head_parameters["value_conv_weight"])
    v_scale = f32(head_parameters["value_scale"])
    v_shift = f32(head_parameters["value_shift"])
    v_hid_w = f32(head_parameters["value_hidden_weight"])
    v_hid_b = f32(head_parameters["value_hidden_bias"])
    v_out_w = f32(head_parameters["value_output_weight"])
    v_out_b = f32(head_parameters["value_output_bias"])

    v = F.conv2d(h, v_conv_w, stride=1, padding=0, bias=None)
    v = torch.relu(_bn(v, v_scale, v_shift, eps))
    v = v.flatten(1)

    v = torch.relu(F.linear(v, v_hid_w, v_hid_b))
    v = torch.tanh(F.linear(v, v_out_w, v_out_b))
    values = v.squeeze(-1)

    return policy_logits, values
