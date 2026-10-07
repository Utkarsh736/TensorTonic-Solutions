import torch
import math

def causal_gqa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Returns the attention tensor with the same shape and dtype as q.
    """
    B, H_q, S, D = q.shape
    _, H_kv, _, _ = k.shape

    G = H_q//H_kv

    q_f = q.float()
    k_f = k.float()
    v_f = v.float()

    q_grouped = q_f.reshape(B, H_kv, G, S, D)
    k_t = k_f.transpose(-2, -1).unsqueeze(2)

    scores = torch.matmul(q_grouped, k_t)/math.sqrt(D)

    # Mask
    mask = torch.triu(torch.ones(S, S, device=q.device), diagonal=1).bool()
    scores = scores.masked_fill(mask, float('-inf'))

    # Softmax
    probs = torch.softmax(scores, dim=-1)
    out = probs@v_f.unsqueeze(2)

    out = out.reshape(B, H_q, S, D)

    return out.to(q.dtype)