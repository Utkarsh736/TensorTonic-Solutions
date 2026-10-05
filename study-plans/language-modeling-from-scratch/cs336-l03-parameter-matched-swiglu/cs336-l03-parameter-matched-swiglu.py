import torch
import math

def parameter_matched_swiglu(
    x: torch.Tensor, w_g: torch.Tensor, w_v: torch.Tensor,
    w_o: torch.Tensor, base_params: int,
) -> dict:
    """
    Returns output, hidden_width, and parameter_count in a dictionary.
    """
    d = x.shape[-1]
    H = w_g.shape[-1]

    h = min(H, max(1, math.floor(base_params/(3*d) + 0.5)))

    w_g_slice = w_g[:, :h]
    w_v_slice = w_v[:, :h]
    w_o_slice = w_o[:h, :]

    x_f = x.float()
    w_g_f = w_g_slice.float()
    w_v_f = w_v_slice.float()
    w_o_f = w_o_slice.float()

    gate = x_f@w_g_f
    silu_gate = torch.nn.functional.silu(gate)
    value = x_f@w_v_f
    mixed = silu_gate * value
    out_f = mixed@w_o_f
    output = out_f.to(x.dtype)

    param_count = 3*d*h

    return {
    "output": output,
    "hidden_width": h,
    "parameter_count": param_count,
}