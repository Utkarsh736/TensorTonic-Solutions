import torch
import math

def attention_scores(q: torch.Tensor, k: torch.Tensor, num_heads: int) -> torch.Tensor:
    """
    Returns scores of shape (batch, heads, query_length, key_length).
    """
    B, S_q, D = q.shape
    S_k = k.shape[1]
    H = num_heads
    dh = D // H

    q_head = q.reshape(B, S_q, H, dh).transpose(1,2)
    k_head = k.reshape(B, S_k, H, dh).transpose(1,2)

    scores = torch.einsum('bhqd,bhkd->bhqk', q_head, k_head)

    return scores/math.sqrt(dh)
